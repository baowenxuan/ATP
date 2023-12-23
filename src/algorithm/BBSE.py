import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient


class BBSEServer(BaseServer):

    def __init__(self, train_datasets, test_datasets, args):

        BaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: BBSEClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: BBSEClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

    def run(self, args):
        self.learn_to_adapt(args)
        self.adapt_and_eval(args)

    def learn_to_adapt(self, args):

        state = deepcopy(self.model.state_dict())

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        confsum = torch.zeros((args.num_labels, args.num_labels)).to(args.device)

        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            conf_mat = client.local_confusion(self.model, args, 'test')

            confsum += conf_mat

        conf = confsum / confsum.sum(dim=0, keepdim=True)

        print(conf)



        self.pinv_conf = torch.pinverse(conf)

    def adapt_and_eval(self, args):
        # current global model
        global_state = deepcopy(self.model.updated_state_dict())

        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.test_clients.items()):
            loss, metric, num_data = client.local_eval(self.model, self.pinv_conf, args, 'test')
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


class BBSEClient(BaseClient):

    def local_confusion(self, model, args, dataset='test'):
        model.eval()

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        conf_mat = torch.zeros((args.num_labels, args.num_labels)).to(self.device)

        for i, (*X, Y) in enumerate(dataloader):
            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            with torch.no_grad():

                logits = model(*X)

                if args.conf_mode == 'hard':
                    pred = logits.argmax(dim=1)
                    for j in range(len(Y)):
                        conf_mat[pred[j], Y[j]] += 1

                elif args.conf_mode == 'soft':
                    dist = torch.softmax(logits, dim=1)
                    for j in range(len(Y)):
                        conf_mat[:, Y[j]] += dist[j]

        return conf_mat  # unnormalized

    def local_eval(self, model, pinv_conf, args, dataset='test'):

        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        for i, (*X, Y) in enumerate(dataloader):
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            with torch.no_grad():
                logits = model(*X)

                if args.conf_mode == 'hard':
                    pred = logits.argmax(dim=1)

                    pred_dist = torch.zeros(logits.shape[1]).to(args.device)
                    for p in pred:
                        pred_dist[p] += 1.0

                    pred_dist = pred_dist / pred_dist.sum()


                elif args.conf_mode == 'soft':
                    pred_dist = torch.softmax(logits, dim=1).mean(dim=0)

                # print(pinv_conf.device, pred_dist.device)
                # print(pred_dist)
                label_dist = torch.matmul(pinv_conf, pred_dist)
                # print(label_dist)

                # print(label_dist)
                label_dist = project_to_simplex(label_dist)

                smoothing = torch.ones_like(label_dist)
                smoothing = smoothing / smoothing.sum()
                eps = 0.001

                label_dist = eps * smoothing + (1 - eps) * label_dist

                bias = torch.log(label_dist)

                logits = logits + bias.unsqueeze(0)

                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data


def project_to_simplex(v):
    """
    Projects a vector onto the probability simplex.

    Args:
    - v: a PyTorch tensor of shape (n,) representing the vector to be projected.

    Returns:
    - A PyTorch tensor of shape (n,) representing the projection of v onto the probability simplex.
    """
    sorted_v, _ = torch.sort(v, descending=True)
    cumsum_v = torch.cumsum(sorted_v, dim=0)
    t = torch.arange(1, v.shape[0] + 1, device=v.device, dtype=v.dtype)
    rho = torch.sum(sorted_v * t > (cumsum_v - 1)) - 1
    theta = (cumsum_v[rho] - 1) / (rho + 1)
    return torch.clamp(v - theta, min=0)
