import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class SHOTServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: SHOTClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: SHOTClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        if args.model == 'cnn':
            self.model.linear5.requires_grad_(False)
        else: # resnet
            self.model.backbone.fc.requires_grad_(False)  # freeze classifier


class SHOTClient(BaseClient):


    def local_eval(self, model, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        featurizer = model.get_featurizer()
        classifier = model.get_classifier()

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            feature = featurizer(*X)
            # print(feature.shape)
            logits = classifier(feature)

            # entropy
            loss_ent = unspv_loss_func(logits, Y)

            # diversity
            pred_dist = torch.softmax(logits, dim=1).mean(dim=0)
            loss_div = torch.sum(pred_dist * torch.log(pred_dist + 1e-9))
            # loss_div = - unspv_loss_func(logits, None)

            if args.shot_beta == 0:
                loss = loss_ent + loss_div

            else:

                # construct PL
                with torch.no_grad():
                    pred = torch.softmax(logits, dim=1)
                    pred = pred / pred.sum(dim=0)

                    prototype = torch.matmul(pred.transpose(1, 0), feature)  # numclass * dim_feature

                    PL = torch.zeros_like(Y)

                    for i in range(len(Y)):
                        dist = (feature[i] - prototype).norm(dim=1)
                        PL[i] = torch.argmin(dist)

                loss_pl = spv_loss_func(logits, PL)
                loss = loss_ent + loss_div + args.shot_beta * loss_pl


            loss.backward()
            # print(model.backbone.bn1.weight)
            optimizer.step()
            # print(model.backbone.bn1.weight)
            # exit()
            optimizer.zero_grad()

            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        # exit()

        return avg_loss, avg_metric, total_examples

