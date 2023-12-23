import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

def dict_max(d):
    return np.max([np.max(v) for v in d.values()])


class create_digits_dataset(Dataset):
    def __init__(self, data_path, channels, is_test, data_holdout):
        self.client_sample_id_cur_dataset = {}
        for part in range(10):
            images, labels = np.load(os.path.join(data_path, 'partitions/train_ls_part{}.pkl'.format(part)), allow_pickle=True)
            if part==0:
                self.images, self.labels = images, labels
            else:
                self.images = np.concatenate([self.images,images], axis=0)
                self.labels = np.concatenate([self.labels,labels], axis=0)
            if is_test:
                sids = np.array(list(range(len(self.images)-len(images), len(self.images))))
                self.client_sample_id_cur_dataset[part] = {"test": sids}
            else:
                sids = np.array(list(range(len(self.images)-len(images), len(self.images))))
                np.random.shuffle(sids)
                split_thresh = int((1-data_holdout)*len(images))
                train_sids, test_sids = sids[:split_thresh], sids[split_thresh:]
                self.client_sample_id_cur_dataset[part] = {"train": train_sids, "test": test_sids}
                
        self.n_samples_cur_dataset = len(self.images)
        self.n_clients_cur_dataset = len(self.client_sample_id_cur_dataset)
        self.channels = channels
        if self.channels==1:
            self.transform = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.channels==3:
            self.transform = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
