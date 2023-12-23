import os
import numpy as np
from torchvision import datasets, transforms


def create_torchvision_dataset(dataset_name='mnist', data_dir='../data'):
    """
    Use dataset given in torchvision
    :param dataset_name: name of the dataset
    :param data_dir: directory of the dataset, e.g., ../data
    :return: train_dataset, test_dataset
    """
    data_dir = os.path.join(data_dir, 'torchvision')

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean and std of mnist
        ])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))])  # mean and std of fmnist

        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean and std of each channel
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'coarse-cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),  # mean and std of each channel
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        test_dataset.targets = sparse2coarse(test_dataset.targets)

    else:
        raise NotImplementedError('It is not a torchvision dataset! Please check the dataset name. ')

    return train_dataset, test_dataset


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def test():
    import torch
    from torch.utils.data import DataLoader
    import time
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    train_dataset, test_dataset = create_torchvision_dataset(dataset_name='cifar100', data_dir='~/data')
    # print(train_dataset[0])
    print(train_dataset[0][0].shape)
    plt.imshow(train_dataset[0][0].permute(1, 2, 0))
    plt.show()


    # stat = {i: 0 for i in range(10)}
    # for i in range(len(train_dataset)):
    #     stat[train_dataset[i][-1]] += 1
    # print(stat)
    #
    # stat = {i: 0 for i in range(10)}
    # for i in range(len(test_dataset)):
    #     stat[train_dataset[i][-1]] += 1
    # print(stat)



    tensor = []
    tik = time.time()
    for i in range(len(train_dataset)):
        tensor.append(train_dataset[i][0])
        # if i == 0:
        #     print(train_dataset[i])
        #     break
        # if i == 50000:
        #     print(train_dataset[i])
    tok = time.time()
    print('Time cost', tok - tik, 's')
    tensor = torch.stack(tensor)
    print(tensor.mean(dim=(0, 2, 3)), tensor.std(dim=(0, 2, 3)))
    print(tensor.std(dim=(2, 3)).mean(dim=0))
    print(len(tensor))

    # dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # tik = time.time()
    # for *X, Y in tqdm(dataloader):
    #     pass
    # tok = time.time()
    # print('Time cost', tok - tik, 's')