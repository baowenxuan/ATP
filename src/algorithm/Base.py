"""
An Abstract Base Class of Federated Learning
"""

import torch
from torch.utils.data import DataLoader

from model import create_model, create_loss, create_metric, create_optimizer
from utils import History


class BaseServer:
    """
    Base Class of Server
    """

    def __init__(self, train_datasets, test_datasets, args):
        # some useful information
        self.num_train_clients = len(train_datasets)
        self.num_test_clients = len(test_datasets)

        self.train_idx2cid = {i: cid for i, cid in enumerate(train_datasets)}
        self.train_cid2idx = {cid: i for i, cid in self.train_idx2cid.items()}

        self.test_idx2cid = {i: cid for i, cid in enumerate(test_datasets)}
        self.test_cid2idx = {cid: i for i, cid in self.test_idx2cid.items()}

        self.gm_rounds = args.gm_rounds

        # history
        self.history = History()
        self.history.concat({
            'train_idx2cid': self.train_idx2cid,
            'train_cid2idx': self.train_cid2idx,
            'test_idx2cid': self.test_idx2cid,
            'test_cid2idx': self.test_cid2idx,
        })


class BaseClient:
    """
    Base Class of Client
    """

    def __init__(self, cid, datasets, args):
        self.cid = cid

        # client local dataset
        self.datasets = datasets  # e.g. ['train', 'test'] or ['train', 'val', 'test']

        # number of training data / testing data / ...
        self.num_data = {key: len(dataset) for key, dataset in datasets.items()}
        self.num_data['all'] = sum([len(dataset) for dataset in datasets.values()])

        # client local dataloaders
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            if key in ['train', ]:
                # for training set, we shuffle the data
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                   num_workers=self.num_workers)

            elif key in ['valid', 'test', ]:
                # for validation and testing set, it is not necessary to shuffle
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   num_workers=self.num_workers)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        (Just a template. )
        """
        model.train()

        avg_loss, avg_metric = float('inf'), 0.0
        num_data = self.num_data[dataset]

        return avg_loss, avg_metric, num_data

    def local_eval(self, model, args, dataset='test'):
        """
        Local Evaluation
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        with torch.no_grad():
            for *X, Y in self.dataloaders[dataset]:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)
                metric = metric_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples
