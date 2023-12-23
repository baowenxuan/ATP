import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient


class TTPBaseServer(BaseServer):
    """
    Base Class for Test-Time Personalization
    """

    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

    def run(self, args):

        # Performance before adaptation
        tqdm.write('No Adaptation: ')
        self.adapt_and_eval(args)

        # Iteratively adapt
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.learn_to_adapt(args)
            self.adapt_and_eval(args)

    def learn_to_adapt(self, args):
        """
        Use training clients' validation data to learn to adapt
        """

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        # current global model
        global_state = deepcopy(self.model.updated_state_dict())
        global_adapt_lrs = deepcopy(self.adapt_lrs)

        accum_adapt_lrs = torch.zeros_like(self.adapt_lrs)

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            loss, metric, num_data = client.local_train(self.model, self.adapt_lrs, args, 'test')

            # the local adaptation rate
            accum_adapt_lrs += self.adapt_lrs

            # reset (1) the model and (2) the adaptation rate
            self.adapt_lrs.copy_(global_adapt_lrs)  # set it back.
            self.model.load_state_dict(global_state, strict=False)

            # some statistics
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # global adaptation rate (used for next round)
        self.adapt_lrs = accum_adapt_lrs / len(selected_cids)

        # print(losses)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Train:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'adapt_lrs': self.adapt_lrs.cpu().numpy(),
            'train_losses': losses,
            'train_metrics': metrics,
            'train_wavg_loss': agg_loss,
            'train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    def adapt_and_eval(self, args):
        # current global model
        global_state = deepcopy(self.model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        if args.verbose:
            print(self.adapt_lrs)

        for cid, client in tqdm(self.test_clients.items()):
            loss, metric, num_data = client.local_eval(self.model, self.adapt_lrs, args, 'test')
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
            'adapt_lrs': self.adapt_lrs.cpu().numpy(),
            'test_losses': losses,
            'test_metrics': metrics,
            'test_wavg_loss': agg_loss,
            'test_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)
