import numpy as np
import matplotlib.pyplot as plt


def print_label_distribution_stat(dataset, num_labels, partition_idxs, visualize=False, resize=0.2):
    num_clients = len(partition_idxs)
    labels = [data[-1] for data in dataset]

    label_dist = np.zeros((num_clients, num_labels), dtype=int)
    for cid, sids in partition_idxs.items():
        for sid in sids:
            label_dist[cid, labels[sid]] += 1

    print('Label Distribution:')
    print(label_dist)

    if visualize:
        x = np.tile(np.arange(num_clients), num_labels)
        y = np.repeat(np.arange(num_labels), num_clients)
        size = label_dist[x, y]

        plt.scatter(x, y, s=size * resize)
        plt.xticks(range(num_clients))
        plt.yticks(range(num_labels))
        plt.xlabel('Clients')
        plt.ylabel('Labels')
        plt.title('Label Distribution')
        plt.show()


def print_quantity_stat(partition_idxs, visualize=False):
    quantities = [len(idxs) for idxs in partition_idxs.values()]
    lo = np.min(quantities)
    lo4 = np.quantile(quantities, 1 / 4)
    md = np.median(quantities)
    hi4 = np.quantile(quantities, 3 / 4)
    hi = np.max(quantities)
    mu = np.mean(quantities)
    sd = np.std(quantities, ddof=1)
    print('Quantity: \n\tQuantiles:', lo, lo4, md, hi4, hi)
    print('\tMean +- Std: %f +- %f' % (mu, sd))

    if visualize:
        plt.hist(quantities)
        plt.title('Quantity Distribution')
        plt.show()
