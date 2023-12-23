import torch
import numpy as np
import random
from copy import deepcopy

from dataset import create_fed_dataset
from algorithm import create_system
from utils import pickle_load, pickle_save

from options import args_parser


def main(args):
    args_backup = deepcopy(args)

    # get dataset
    train_datasets, test_datasets = create_fed_dataset(args)

    # get system
    server = create_system(train_datasets, test_datasets, args)

    # run experiments
    server.run(args)

    content = {
        'args': args_backup,
        'history': server.history.data,
    }

    if args.history_path != 'none':
        pickle_save(content, args.history_path, mode='ab')

    if args.save_model_path != 'none':
        pickle_save(server.model.state_dict(), args.save_model_path, mode='wb')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # the following three lines seem not necessary
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    main(args)
