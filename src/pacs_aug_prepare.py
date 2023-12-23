import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle
from tqdm import tqdm

from dataset import PACS_Aug
from utils import pickle_save, pickle_load
from partition import create_partition
from options import args_parser





def main(args):

    print(args.domain_path)

    if not os.path.exists(args.domain_path):
        datasets = PACS_Aug(root=args.data_dir, test_idx=args.partition_seed)

        # work with the dataset itself.

        tosave = []

        for dataset in datasets:
            X, Y = [], []
            for i in tqdm(range(len(dataset))):
                X.append(dataset[i][0])
                Y.append(dataset[i][1])

            X = torch.stack(X)
            Y = torch.LongTensor(Y)

            print(X.dtype, X.shape, Y.dtype, Y.shape)
            tosave.append([X, Y])

        pickle_save(obj=tosave,
                    file=args.domain_path, mode='wb')


    datasets = []

    mats = pickle_load(args.domain_path)

    for (X, Y) in mats:
        datasets.append(TensorDataset(X, Y))


    # work with the data partition

    all_train_partitions = {}
    train_environments = {}

    all_test_partitions = {}
    test_environments = {}

    all_key = 0

    for environment in range(4):
        dataset = datasets[environment]

        # get partition of datasets
        args.num_clients = 7
        args.num_labels = 7
        args.data_holdout = 0.2

        if environment == args.partition_seed:  # testing:
            args.client_holdout = 1.0  # all test

        else:  # training
            args.client_holdout = 0.0  # all train

        train_partitions, test_partitions, partition_idxs = create_partition([dataset, ], args)

        for key in train_partitions:
            all_train_partitions[all_key] = train_partitions[key]
            train_environments[all_key] = environment
            all_key += 1

        for key in test_partitions:
            all_test_partitions[all_key] = test_partitions[key]
            test_environments[all_key] = environment
            all_key += 1

    print(all_train_partitions.keys(), all_test_partitions.keys())

    assert len(set(all_train_partitions).union(set(all_test_partitions))) == 28  # there should be 28 clients.

    pickle_save(obj=[all_train_partitions, all_test_partitions, train_environments, test_environments],
                file=args.partition_path, mode='wb')


def set_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.partition_seed)
    main(args)