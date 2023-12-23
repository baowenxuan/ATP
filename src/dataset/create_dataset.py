from torch.utils.data import ConcatDataset, Subset, TensorDataset

from utils import pickle_load
import os
from .torchvision_dataset import create_torchvision_dataset
from .digit_dataset import create_digits_dataset
import torchvision.transforms as transforms
from .PACS_Aug import PACS_Aug


def create_dataset(dataset_name, data_dir):
    """
    Create the dataset and its partition
    :param args:
    :return:
    """
    torchvision_dataset_names = [
        'mnist',
        'fmnist',
        'cifar10',
        'cifar100',
        'coarse-cifar100',
    ]

    if dataset_name in torchvision_dataset_names:
        datasets = create_torchvision_dataset(dataset_name=dataset_name, data_dir=data_dir)
    else:
        raise NotImplementedError('Unknown dataset!')
    return datasets


def dict_add(values, num):
    dict_new = {}
    for key, value in values.items():
        dict_new[key] = value + num
    return dict_new


def update_client_sample(dataset, is_test, n_clients, n_samples, train_client_sample_id, test_client_sample_id):
    client_sample_id_cur_dataset, n_clients_cur_dataset, n_samples_cur_dataset = dataset.client_sample_id_cur_dataset, dataset.n_clients_cur_dataset, dataset.n_samples_cur_dataset
    if is_test:
        test_client_sample_id = {key + n_clients: dict_add(values, n_samples) for key, values in
                                 client_sample_id_cur_dataset.items()}
    else:
        client_sample_id_cur_dataset = {key + n_clients: dict_add(values, n_samples) for key, values in
                                        client_sample_id_cur_dataset.items()}
        train_client_sample_id = {**train_client_sample_id, **client_sample_id_cur_dataset}
    n_clients, n_samples = n_clients + n_clients_cur_dataset, n_samples + n_samples_cur_dataset
    return train_client_sample_id, test_client_sample_id, n_clients, n_samples


def create_dataset_domain_shift(data_dir, args):
    """
    Create the dataset and its partition
    :param args:
    :return:
    """
    dataset_names = [
        'mnist',
        'svhn',
        'usps',
        'synth_digits',
        'mnist_m',
    ]
    train_client_sample_id, test_client_sample_id = {}, {}
    n_clients, n_samples = 0, 0
    seed, data_holdout = args.partition_seed, args.data_holdout
    # MNIST
    is_test = seed % 5 == 0
    mnist_dataset = create_digits_dataset(data_path=os.path.join(data_dir, 'MNIST'), channels=1, \
                                          is_test=is_test, data_holdout=data_holdout)
    train_client_sample_id, test_client_sample_id, n_clients, n_samples = update_client_sample(mnist_dataset, is_test,
                                                                                               n_clients, n_samples,
                                                                                               train_client_sample_id,
                                                                                               test_client_sample_id)
    # SVHN
    is_test = seed % 5 == 1
    svhn_dataset = create_digits_dataset(data_path=os.path.join(data_dir, 'SVHN'), channels=3, \
                                         is_test=is_test, data_holdout=data_holdout)
    train_client_sample_id, test_client_sample_id, n_clients, n_samples = update_client_sample(svhn_dataset, is_test,
                                                                                               n_clients, n_samples,
                                                                                               train_client_sample_id,
                                                                                               test_client_sample_id)

    # USPS
    is_test = seed % 5 == 2
    usps_dataset = create_digits_dataset(data_path=os.path.join(data_dir, 'USPS'), channels=1, \
                                         is_test=is_test, data_holdout=data_holdout)
    train_client_sample_id, test_client_sample_id, n_clients, n_samples = update_client_sample(usps_dataset, is_test,
                                                                                               n_clients, n_samples,
                                                                                               train_client_sample_id,
                                                                                               test_client_sample_id)

    # Synth Digits
    is_test = seed % 5 == 3
    synth_dataset = create_digits_dataset(data_path=os.path.join(data_dir, 'SynthDigits'), channels=3, \
                                          is_test=is_test, data_holdout=data_holdout)
    train_client_sample_id, test_client_sample_id, n_clients, n_samples = update_client_sample(synth_dataset, is_test,
                                                                                               n_clients, n_samples,
                                                                                               train_client_sample_id,
                                                                                               test_client_sample_id)

    # MNIST-M
    is_test = seed % 5 == 4
    mnistm_dataset = create_digits_dataset(data_path=os.path.join(data_dir, 'MNIST_M'), channels=3, \
                                           is_test=is_test, data_holdout=data_holdout)
    train_client_sample_id, test_client_sample_id, n_clients, n_samples = update_client_sample(mnistm_dataset, is_test,
                                                                                               n_clients, n_samples,
                                                                                               train_client_sample_id,
                                                                                               test_client_sample_id)

    datasets = [mnist_dataset, svhn_dataset, usps_dataset, synth_dataset, mnistm_dataset]
    return datasets, train_client_sample_id, test_client_sample_id


def load_processed_dataset(path):
    print('Load Processed Data:', path)
    obj = pickle_load(path)
    X = obj['X']
    Y = obj['Y']
    dataset = TensorDataset(X, Y)
    return dataset


def create_fed_dataset(args, config=None):
    dataset_name = args.dataset
    data_dir = args.data_dir
    partition_path = args.partition_path

    if args.dataset in ['pacs', 'vlcs', 'pacs_aug']:  # bypass

        datasets = []

        mats = pickle_load(args.domain_path)

        for (X, Y) in mats:
            datasets.append(TensorDataset(X.cuda(), Y.cuda()))

        # datasets = PACS(root=args.data_dir)

        all_train_partitions, all_test_partitions, train_environments, test_environments = pickle_load(partition_path)

        train_datasets = {
            cid: {part: Subset(datasets[train_environments[cid]], indices)
                  for part, indices in all_train_partitions[cid].items()}
            for cid in all_train_partitions
        }

        test_datasets = {
            cid: {part: Subset(datasets[test_environments[cid]], indices)
                  for part, indices in all_test_partitions[cid].items()}
            for cid in all_test_partitions
        }

        return train_datasets, test_datasets

    if args.corruption == "none":
        datasets = create_dataset(dataset_name, data_dir)
        dataset = ConcatDataset(datasets)
    elif args.corruption == "domain":
        datasets, train_client_sample_id, test_client_sample_id = create_dataset_domain_shift(args.data_dir, args)
        dataset = ConcatDataset(datasets)
    else:
        dataset = load_processed_dataset(args.corruption_path)
    print(len(dataset))
    if args.corruption != "domain":
        train_client_sample_id, test_client_sample_id = pickle_load(partition_path)
    train_datasets = {
        cid: {part: Subset(dataset, indices) for part, indices in sids.items()} for cid, sids in
        train_client_sample_id.items()
    }

    test_datasets = {
        cid: {part: Subset(dataset, indices) for part, indices in sids.items()} for cid, sids in
        test_client_sample_id.items()
    }

    return train_datasets, test_datasets
