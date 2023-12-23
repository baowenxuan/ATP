import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer
from utils.third_party import aug_digit, aug_cifar, aug_pacs

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class MEMOServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        prior = args.prior_strength / (args.prior_strength + 1)

        # self.model.change_bn(mode='prior', prior=prior)

        self.model.eval()


class MEMOClient(BaseClient):

    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        if args.dataset == 'cifar10':
            self.tr_pre = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1 / 0.2470, 1 / 0.2435, 1 / 0.2616)),
                transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
                transforms.ToPILImage()
            ])
            self.tr_post = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
            ])
            self.aug = aug_cifar

        elif args.dataset == 'digit':
            self.tr_pre = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                transforms.ToPILImage()
            ])
            self.tr_post = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # mean and std of each channel
            ])
            self.aug = aug_digit

        elif args.dataset == 'cifar100':
            self.tr_pre = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1 / 0.2673, 1 / 0.2546, 1 / 0.2762)),
                transforms.Normalize((-0.5071, -0.4866, -0.4409), (1, 1, 1)),
                transforms.ToPILImage()
            ])
            self.tr_post = transforms.Compose([
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
            ])
            self.aug = aug_cifar

        elif args.dataset == 'pacs_aug':
            self.tr_pre = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
                transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
                transforms.ToPILImage()
            ])
            self.tr_post = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std of each channel
            ])
            self.aug = aug_pacs

        else:
            raise NotImplementedError('MEMO requires data in its original space')
        self.batch_size = 1

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            if key in ['train', ]:
                # for training set, we shuffle the data, we drop too small batch in the training if necessary
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                   num_workers=self.num_workers)

            elif key in ['valid', 'test', ]:
                # for testing set, it is not necessary to shuffle
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   num_workers=self.num_workers)


    def adapt_single(self, model, image, optimizer, args):
        model.eval()
        image = self.tr_pre(image)
        inputs = [self.tr_post(self.aug(image)) for _ in range(args.memo_aug_size)]
        inputs = torch.stack(inputs).to(self.device)

        # print(inputs.shape)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss, logits = marginal_entropy(outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    def local_eval(self, model, args, dataset='test'):

        spv_loss_func = create_loss('ce')
        metric_func = create_metric('acc')
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]

        state = deepcopy(model.state_dict())

        for *X, Y in dataloader:
            model.load_state_dict(state)

            image = X[0][0]  # get the first sample (only sample)
            Y = Y.to(self.device)

            self.adapt_single(model, image, optimizer, args)

            model.eval()

            X = [x.to(self.device) for x in X]

            # print(X[0].shape)

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        # print(avg_metric)

        return avg_loss, avg_metric, total_examples





def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

