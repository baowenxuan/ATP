import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient


class SurgicalServer(BaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: SurgicalClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: SurgicalClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

    def run(self, args):
        self.learn_to_adapt(args)
        self.adapt_and_eval(args)

    def learn_to_adapt(self, args):
        """
        Learn which layers to adapt. We also use the adaptation rates ...
        """

        state = deepcopy(self.model.state_dict())

        if args.surgical_metric == 'valid':

            modes = ['block1', 'block2', 'block3', 'block4', 'last_layer']
            accs = []
            for mode in modes:
                self.model.load_state_dict(state)
                self.model.surgical(mode)
                self.model.train()

                # sample a subset of clients
                selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
                selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

                weights = []  # weights (importance) for each client
                losses = []  # local testing losses
                metrics = []  # local testing metrics (accuracies)

                for cid in tqdm(selected_cids):

                    client = self.train_clients[cid]
                    loss, metric, num_data = client.local_eval(self.model, args, 'test')

                    # some statistics
                    weights.append(num_data)
                    losses.append(loss)
                    metrics.append(metric)

                    self.model.load_state_dict(state)

                # eval loss and metric
                agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
                agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
                # tqdm.write('\t Train:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

                print(mode, agg_metric)
                accs.append(agg_metric)

            mode = modes[np.argmax(accs)]
            print(mode)
            self.model.load_state_dict(state)
            self.model.surgical(mode)

        else:
            raise NotImplementedError

    def adapt_and_eval(self, args):
        # current global model
        global_state = deepcopy(self.model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.test_clients.items()):
            loss, metric, num_data = client.local_eval(self.model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # reset the model (the adaptation rate is not update, do not need to reset)
            self.model.load_state_dict(global_state, strict=False)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_losses': losses,
            'test_metrics': metrics,
            'test_wavg_loss': agg_loss,
            'test_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class SurgicalClient(BaseClient):

    def adapt_one_step(self, model, X, Y, unspv_loss_func, args):
        model.eval()

        logits = model(*X)

        loss = unspv_loss_func(logits, Y)

        loss.backward()

        model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= args.lm_lr * g

        model.zero_grad()

        model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.

        return unspv_grad

    def local_eval(self, model, args, dataset='test'):

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        state = deepcopy(model.state_dict())

        for i, (*X, Y) in enumerate(dataloader):
            model.load_state_dict(state)

            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1. unsupervised adaptation

            model.train()

            self.adapt_one_step(model, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation

            model.eval()

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
