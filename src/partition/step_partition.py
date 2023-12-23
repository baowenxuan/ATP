"""
Step Partition
"""

import numpy as np

from .utils import get_labels

import itertools


def step_partition(dataset, num_labels, num_clients, num_major, alpha):
    """
    :param dataset: Dataset
    :param num_clients: number of clients ()
    :param alpha: concentration score. Larger alpha -> more IID
    :return:
    """

    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    # label skewness: control each client's label distribution (separately)
    prior = num_samples_per_label / num_samples_per_label.sum()

    if alpha == float('inf'):
        matrix = np.zeros((num_clients, num_labels))
        alpha = 2
    else:
        matrix = np.ones((num_clients, num_labels))

    if num_clients == 300 and num_labels == 10 and num_major == 2:  # CIFAR-10 experiments
        for cid, label_ids in enumerate(itertools.product(range(num_labels), repeat=num_major)):
            for label_id in label_ids:
                matrix[cid * 3:(cid + 1) * 3, label_id] += (alpha - 1)

    elif num_clients == 10 and num_labels == 10 and num_major == 2:  # Digits experiments
        for cid in range(10):
            matrix[cid, cid] += (alpha - 1)
            matrix[cid, (cid + 1) % 10] += (alpha - 1)

    elif num_clients == 7 and num_labels == 7 and num_major == 2:  # PACS experiments
        for cid in range(7):
            matrix[cid, cid] += (alpha - 1)
            matrix[cid, (cid + 1) % 7] += (alpha - 1)

    elif num_clients == 300 and num_labels == 100:  # CIFAR-100 experiments
        for i in range(3):
            label_gap = 1 + 2 * i
            client_init = i * 100
            for label_init in range(100):
                cid = client_init + label_init
                for j in range(num_major):
                    label_id = (label_init + label_gap * j) % 100
                    matrix[cid, label_id] += (alpha - 1)

    else:
        raise NotImplementedError

    # normalizing matrix
    matrix = matrix / matrix.sum(axis=0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.5).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for cid in range(num_clients):
        idxs = []
        for label in range(num_labels):
            idxs.append(idxs_by_label[label][cumulate[cid, label]:cumulate[cid + 1, label]])

        partition_idxs[cid] = np.concatenate(idxs)

    return partition_idxs
