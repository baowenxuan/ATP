import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle

from partition.step_partition import step_partition

def main(root):

    for ds_name in ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']:

        # load data

        for part in range(10):
            path = os.path.join(root, ds_name, 'partitions/train_part{}.pkl'.format(part))
            images, labels = np.load(path, allow_pickle=True)

            if part == 0:
                all_images, all_labels = images, labels

            else:
                all_images = np.concatenate([all_images, images], axis=0)
                all_labels = np.concatenate([all_labels, labels], axis=0)

        print(all_images, all_labels)



        ds = TensorDataset(torch.Tensor(all_images), torch.LongTensor(all_labels))

        # exit()

        partition_idxs = step_partition(dataset=ds, num_labels=10, num_clients=10, num_major=2, alpha=16)

        for part in range(10):
            path = os.path.join(root, ds_name, 'partitions/train_ls_part{}.pkl'.format(part))

            idxs = partition_idxs[part]
            np.random.shuffle(idxs)
            images = all_images[idxs]
            labels = all_labels[idxs]

            with open(path, 'wb') as f:
                pickle.dump((images, labels), f, pickle.HIGHEST_PROTOCOL)







if __name__ == '__main__':
    np.random.seed(0)
    root = '../../data/'
    main(root)
