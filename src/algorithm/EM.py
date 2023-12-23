import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class EMServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: EMClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: EMClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.eval()


class EMClient(BaseClient):

    def local_eval(self, model, args, dataset='test'):

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
                pred0 = torch.softmax(logits, dim=1)

                pred = pred0.clone()

                for ep in range(args.em_epochs):
                    # get new prior distribution
                    label_dist = pred.mean(dim=0)

                    # get prior-adjusted prediction
                    pred = pred0 * label_dist
                    pred = pred / pred.sum(dim=1, keepdim=True)

                logits = torch.log(pred)

                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
