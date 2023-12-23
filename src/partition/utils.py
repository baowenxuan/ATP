import numpy as np
from torch.utils.data import Subset


def get_labels(dataset, num_labels):
    """
    Get label-related information from dataset
    :param dataset: torch.utils.data.Dataset
    :return:
        labels: numpy.array the labels of data in dataset
        idxs_by_class: dictionary {label:idxs}, where idxs is a (shuffled) list of data idxs with given label
        num_labels: int, how many labels
        num_samples_per_label: numpy.array, how many samples of each label
    """
    labels = [Y for *X, Y in dataset]
    labels = np.array(labels)

    idxs_by_class = {}
    num_samples_per_label = np.zeros(num_labels, dtype=int)

    for label in range(num_labels):
        idxs = np.where(labels == label)[0]  # np.where returns a tuple with length 1
        np.random.shuffle(idxs)
        idxs_by_class[label] = idxs
        num_samples_per_label[label] = len(idxs)

    return labels, idxs_by_class, num_samples_per_label


def stratified_split(dataset, idxs, num_labels, part_rate_dict):
    """
    Stratified Split
    Split a Data subset (dataset, idxs) to multiple parts,
    each with a given rate, while controlling the label distribution
    """

    if idxs is None:
        subdataset = dataset
        idxs = np.arange(len(dataset))
    else:
        subdataset = Subset(dataset, idxs)

    labels, idxs_by_label, num_samples_per_label = get_labels(subdataset, num_labels)
    idx2part = {idx:part for idx, part in enumerate(part_rate_dict)}
    part2idx = {part:idx for idx, part in idx2part.items()}
    num_parts = len(part_rate_dict)

    part_vector = np.zeros(num_parts)
    for part, rate in part_rate_dict.items():
        part_vector[part2idx[part]] = rate
    matrix = np.tile(part_vector, (num_labels, 1)).transpose(1, 0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.49).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for pid in range(num_parts):
        subidxs = []
        for label in range(num_labels):
            subidxs.append(idxs_by_label[label][cumulate[pid, label]:cumulate[pid + 1, label]])

        partition_idxs[idx2part[pid]] = idxs[np.concatenate(subidxs)]

    return partition_idxs
