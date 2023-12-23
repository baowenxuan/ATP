import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class T3AServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: T3AClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: T3AClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.eval()


class T3AClient(BaseClient):

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s))))
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def forward(self, x, adapt=False):
        # if not self.hparams['cached_loader']:
        #     z = self.featurizer(x)
        # else:
        #     z = x

        z = self.featurizer(x)

        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        # print(z.shape)
        # print(torch.nn.functional.normalize(weights, dim=0).shape)
        # print(z)
        # print(torch.nn.functional.normalize(weights, dim=0))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def local_eval(self, model, args, dataset='test'):
        self.featurizer = model.get_featurizer()
        self.classifier = model.get_classifier()

        # print('ca', self.classifier.weight)

        self.num_classes = args.num_labels

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3p_filter_k

        self.softmax = torch.nn.Softmax(-1)

        # below is evaluation

        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            with torch.no_grad():
                logits = self.forward(*X, adapt=True)

                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples






@torch.jit.script
def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
