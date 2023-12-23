"""
FedAvg (Federated Averaging)

Reference:
    Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas:
    Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017: 1273-1282
Implementation:
    https://github.com/pliang279/LG-FedAvg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient


class FedAvgServer(BaseServer):
    """
    Server of FedAvg
    """

    def __init__(self, train_datasets, test_datasets, args):
        super(FedAvgServer, self).__init__(train_datasets, test_datasets, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # init clients
        self.train_clients = {cid: FedAvgClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: FedAvgClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # model
        self.model = create_model(args)

    def run(self, args):
        """
        Run the training and testing pipeline
        """

        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        """
        Train for one communication round
        """
        # current global model
        global_state = deepcopy(model.updated_state_dict())

        # tensors = []  # local model parameters

        next_state = None

        weights = []  # weights (importance) for each client
        losses = []  # training losses for local models (LMs)
        metrics = []  # training metrics (accuracies) for local models (LMs)

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        # iterate randomly selected honest clients
        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            model.load_state_dict(global_state, strict=False)  # start from global model

            loss, metric, num_data = client.local_train(model, args, 'train')
            local_state = model.updated_state_dict()

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # accumulate weights

            if next_state is None:
                next_state = deepcopy(local_state)
                for k in next_state.keys():
                    next_state[k] = torch.mul(local_state[k], num_data)
            else:
                for k in next_state.keys():
                    next_state[k] += torch.mul(local_state[k], num_data)

        # train loss and metric
        sum_weight = sum(weights)
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum_weight
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum_weight
        tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        # aggregate
        for k in next_state.keys():
            next_state[k] = torch.div(next_state[k], sum_weight)

        model.load_state_dict(next_state)

        log_dict = {
            'train_selected_idxs': selected_idxs,
            'train_selected_cids': selected_cids,
            'train_losses': losses,
            'train_metrics': metrics,
            'train_wavg_loss': agg_loss,
            'train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    def eval_part(self, model, args):
        """
        Evaluate the global model with unseen data on participating client (source clients)
        """
        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.train_clients.items()):
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_part_losses': losses,
            'test_part_metrics': metrics,
            'test_part_wavg_loss': agg_loss,
            'test_part_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    def eval_unpart(self, model, args):
        """
        Evaluate the global model with unseen client (target clients)
        """
        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.test_clients.items()):
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'test_unpart_losses': losses,
            'test_unpart_metrics': metrics,
            'test_unpart_wavg_loss': agg_loss,
            'test_unpart_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class FedAvgClient(BaseClient):
    """
    Client of FedAvg
    """

    def __init__(self, cid, datasets, args):
        super(FedAvgClient, self).__init__(cid, datasets, args)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs
        batch_size = args.batch_size

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:

                # Drop the last batch if necessary
                # e.g. when using batch norm
                if model.drop_last and len(Y) < batch_size:
                    continue

                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # record the loss and accuracy
                    num_examples = len(X[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
