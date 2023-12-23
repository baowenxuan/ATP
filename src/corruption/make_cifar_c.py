import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import os

from .distortions import distortions, test_distortions


def make_cifar10_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CIFAR-10-C
        partition_idxs: dictionary of (client index : list of sample indices)
        is_train: whether each client is source client or target client
        data_dir: root of torchvision dataset
        mode: 'iid' or 'ood'.
            'iid' uses the same 15 distortions for both source and target clients
            'ood' uses 15 distortions for source client and 4 additional distortions for target clients
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cifar_c[sid] = x
            labels[sid] = y
    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


def make_cifar100_c(partition_idxs, is_train, data_dir='../data', mode="none"):
    """
    Generate TensorDataset of CIFAR-100-C
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),  # no normalization
    ])

    post_transform = transforms.Compose([
        transforms.ToPILImage(),  # this is important
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_dataset, test_dataset])

    cifar_c, labels = [None] * len(dataset), [None] * len(dataset)

    corruption = {}

    for cid, sids in tqdm(partition_idxs.items()):
        if is_train[cid]:
            distortion, severity = random_distortion(mode="train")
        else:
            distortion, severity = random_distortion(mode=mode)
        corruption[cid] = (distortion.__name__, severity)
        for sid in sids:
            x, y = dataset[sid]
            x = distortion(x, severity=severity)  # add distortion
            x = np.uint8(x)  # convert back to original space
            x = post_transform(x)
            cifar_c[sid] = x
            labels[sid] = y
    assert None not in cifar_c
    assert None not in labels

    cifar_c = torch.stack(cifar_c)
    labels = torch.LongTensor(labels)

    return cifar_c, labels, corruption


def random_distortion(range_severity=(1, 6), mode='train'):
    if mode == "train" or mode == "iid":
        selected_distortions = distortions
    elif mode == "ood":
        selected_distortions = test_distortions
    else:
        raise NotImplementedError

    num_distortions = len(selected_distortions)
    i = np.random.randint(num_distortions)  # randomly choose a distortion
    s = np.random.randint(low=range_severity[0], high=range_severity[1])  # randomly choose severity from 1 to 5
    return selected_distortions[i], s


def test():
    partition_idxs = {
        0: [*range(1)],
        1: [*range(1, 2)]
    }
    make_cifar10_c(partition_idxs)
