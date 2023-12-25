"""
Partition a centralized dataset to clients in FL
"""
import numpy as np

from dataset import create_dataset, shapes_out  # , create_dataset_natural_shift
from partition import create_partition
from corruption import make_cifar10_c, make_cifar100_c
from utils import pickle_save
from options import args_parser
from torch.utils.data import ConcatDataset


# from visual import visualize_label_distribution


def main(args):
    """
    :return: client_sample_id
    {cid: {'train': list of sample ids}}
    """

    # get dataset
    datasets = create_dataset(args.dataset, args.data_dir)
    # get partition of datasets
    *partitions, partition_idxs = create_partition(datasets, args)

    # check correctness
    num_before = sum([len(dataset) for dataset in datasets])

    all_sample = set()

    for partition in partitions:  # train clients or test clients
        for cid, datasets in partition.items():  # which client
            for part, idxs in datasets.items():  # which dataset
                all_sample = all_sample.union(set(idxs))

    num_after = len(all_sample)
    assert num_before == num_after

    pickle_save(obj=partitions, file=args.partition_path, mode='wb')

    # generate corrupted dataset
    if args.corruption == "ood" or args.corruption == "iid":

        is_train = {}

        train_clients, test_clients = partitions

        for cid in train_clients:
            is_train[cid] = True

        for cid in test_clients:
            is_train[cid] = False

        if args.dataset == 'cifar10':
            cifar_c, labels, corruption = make_cifar10_c(partition_idxs, is_train, data_dir=args.data_dir,
                                                         mode=args.corruption)

        elif args.dataset == 'cifar100':
            cifar_c, labels, corruption = make_cifar100_c(partition_idxs, is_train, data_dir=args.data_dir,
                                                          mode=args.corruption)

        obj = {
            'X': cifar_c,
            'Y': labels,
            'corruption': corruption,
        }

        pickle_save(obj=obj, file=args.corruption_path, mode='wb')


def set_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.partition_seed)
    main(args)
