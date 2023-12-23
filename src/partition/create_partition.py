import numpy as np
from torch.utils.data import Subset, ConcatDataset

from .step_partition import step_partition

from .stat import print_quantity_stat, print_label_distribution_stat


def create_partition(datasets, args):
    num_clients = args.num_clients
    num_labels = args.num_labels
    partition_config = args.partition

    data_holdout = args.data_holdout
    client_holdout = args.client_holdout

    dataset = ConcatDataset(datasets)

    partition_idxs = partition(dataset, num_labels, num_clients, partition_config)

    print_label_distribution_stat(dataset, num_labels, partition_idxs, visualize=args.visualize, resize=0.2)
    print_quantity_stat(partition_idxs)
    # split (1) training-testing clients and (2) training-testing samples for training clients

    # 3.1. shuffle the clients
    client_ids = list(partition_idxs.keys())
    np.random.shuffle(client_ids)

    # 3.2. let (1 - client_holdout) * 100% of them be training clients
    split_pivot = round((1 - args.client_holdout) * args.num_clients)
    train_client_sample_id = {}
    for i in range(split_pivot):
        cid = client_ids[i]
        idxs = partition_idxs[cid]
        np.random.shuffle(idxs)

        # let (1 - data_holdout) * 100% of samples be training set
        # and the remaining data_holdout * 100% be testing set
        sample_pivot = round((1 - args.data_holdout) * len(idxs))
        train_client_sample_id[cid] = {
            'train': idxs[:sample_pivot],
            'test': idxs[sample_pivot:],
        }

    # 3.3. let the remaining client_holdout * 100% be testing clients
    test_client_sample_id = {}
    for i in range(split_pivot, args.num_clients):
        cid = client_ids[i]
        idxs = partition_idxs[cid]
        np.random.shuffle(idxs)

        # let all samples be testing set
        test_client_sample_id[cid] = {
            'test': idxs,
        }

    return train_client_sample_id, test_client_sample_id, partition_idxs


def partition(dataset, num_labels, num_clients, partition_config):
    """
    Partition a dataset to several clients. However, there is no train-test split or sampling.
    """
    # parse partition method and parameters
    alg, *params = partition_config.split('_')

    # partition
    if alg == 'step':
        num_major = int(params[0])
        alpha = float(params[1])
        partition_idxs = step_partition(dataset, num_labels, num_clients, num_major, alpha)

    elif alg == 'stratified':
        num_major = 2
        alpha = 1.0
        partition_idxs = step_partition(dataset, num_labels, num_clients, num_major, alpha)

    else:
        raise NotImplementedError('Unknown data partition algorithm. ')

    return partition_idxs
