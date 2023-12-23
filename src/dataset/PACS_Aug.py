import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 4  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_idx):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for i, environment in enumerate(environments):
            path = os.path.join(root, environment)
            print(i, end=' ')
            env_dataset = TransformWrapper(ImageFolder(path), train=(i != test_idx))
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = len(self.datasets[-1].classes)


class TransformWrapper:

    def __init__(self, dataset, train=True):
        self.dataset = dataset

        self.classes = dataset.classes

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print('use augmentation')

        else:

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print('NO augmentation')

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.dataset)


class PACS_Aug(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_idx):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_idx=test_idx)
