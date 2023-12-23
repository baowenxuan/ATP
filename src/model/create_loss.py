import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def create_loss(name='ce'):
    """
    loss function must be differentiable
    """
    if name == 'ce':  # cross entropy loss
        return F.cross_entropy
    elif name == 'ent':  # entropy loss, y is not used
        return lambda x, y: -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)
