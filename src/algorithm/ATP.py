import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d

from .Base import BaseServer, BaseClient
from .TTPBase import TTPBaseServer


class ATPServer(TTPBaseServer):
    """
    A class for debugging
    """

    def __init__(self, train_datasets, test_datasets, args):
        TTPBaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
        # which has identical behavior, but support taking gradient
        self.model.eval()

        num_ar = len([name for name, params in self.model.named_parameters() if params.requires_grad])
        print('Dimension of Adapt Rate:', num_ar)

        args.idx_params = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' not in name]
        args.idx_stats = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' in name]

        print('  - Params:', len(args.idx_params))
        print('  - Stats:', len(args.idx_stats))

        if args.verbose:
            print([name for name, params in self.model.named_parameters() if params.requires_grad])

        self.adapt_lrs = torch.zeros(len(self.model.trainable_parameters())).to(args.device)


class ATPClient(BaseClient):
    """
    A class for debug
    """

    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        self.lr = args.lm_lr  # the learning rate of adaptation rates

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):

        model.eval()

        logits = model(*X)

        loss = unspv_loss_func(logits, Y)

        loss.backward()

        model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= adapt_lrs[i] * g

        model.zero_grad()

        model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.

        return unspv_grad

    def local_train(self, model, adapt_lrs, args, dataset='test'):

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        state = deepcopy(model.state_dict())

        for *X, Y in dataloader:
            model.load_state_dict(state)

            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1. unsupervised adaptation

            unspv_grad = self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation

            model.eval()

            logits = model(*X)
            spv_loss = spv_loss_func(logits, Y)

            spv_grad = torch.autograd.grad(spv_loss, model.trainable_parameters())

            # 3. update the adaptation rate
            with torch.no_grad():

                # manual resize

                if args.grad_norm == 'none':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()

                elif args.grad_norm == 'numel':
                    g = torch.zeros_like(adapt_lrs)
                    l = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()

                    g /= l

                elif args.grad_norm == 'sqrt_numel':
                    g = torch.zeros_like(adapt_lrs)
                    l = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()

                    g /= torch.sqrt(l)

                elif args.grad_norm == 'manual':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                        elif i in args.idx_stats:
                            g[i] += 100 * (g1 * g2).sum()

                elif args.grad_norm == 'params_only':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                            g[i] /= g1.numel()

                elif args.grad_norm == 'stats_only':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_stats:
                            g[i] += (g1 * g2).sum()
                            g[i] /= g1.numel()

                else:
                    raise NotImplementedError

                adapt_lrs += self.lr * g

            with torch.no_grad():
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data

    def local_eval(self, model, adapt_lrs, args, dataset='test'):

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

            self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

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
