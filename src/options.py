import os
import argparse
import torch

from dataset import shapes_in, shapes_out


def args_parser():
    parser = argparse.ArgumentParser()

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Dataset and Partition Config
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--dataset', type=str, default='cifar10',
                        # choices=['cifar10', 'cifar100', 'digit', 'pacs_aug'],
                        help='dataset name')

    parser.add_argument('--num_clients', type=int, default=100,
                        help='number of clients')

    parser.add_argument('--partition', type=str, default='step_2_inf',
                        help='how to partition dataset to clients, in format ${method}_${parameters}')

    parser.add_argument('--data_holdout', type=float, default=0.2,
                        help='hold-out rate of data')

    parser.add_argument('--client_holdout', type=float, default=0.2,
                        help='hold-out rate of clients')

    parser.add_argument('--partition_seed', type=int, default=0,
                        help='pre-defined data partition for each client')

    parser.add_argument('--corruption', type=str, default="none",
                        help='none | iid | ood | domain')

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Model Training
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--model', type=str, default='resnet18',
                        help='federated learning model')

    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'bce'],
                        help='loss function')

    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'bacc'],
                        help='metric function')

    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='the federated learning algorithm')

    # Global model

    parser.add_argument('--gm_opt', type=str, default='sgd',
                        help='global model optimizer')

    parser.add_argument('--gm_lr', type=float, default=1.0,
                        help='learning rate of global model optimizer')

    parser.add_argument('--gm_rounds', type=int, default=100,
                        help='number of global communication rounds')

    parser.add_argument('--part_rate', type=float, default=1.0,
                        help='client participation rate in each communication rounds')

    # Local model

    parser.add_argument('--lm_opt', type=str, default='sgd',
                        help='local model optimizer')

    parser.add_argument('--lm_lr', type=float, default=0.1,
                        help='learning rate of the local model optimizer')

    parser.add_argument('--lm_epochs', type=int, default=1,
                        help='number of local training epochs, each epoch iterates the local dataset once')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Model Adaptation
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--prior_strength', type=float, default=16,
                        help='prior stength for batch norm layers')

    parser.add_argument('--memo_aug_size', type=int, default=16,
                        help='augmentation size for memo')

    parser.add_argument('--t3p_filter_k', type=int, default=-1,
                        help='filter out')

    parser.add_argument('--layers_to_adapt', type=str, default='all',
                        help='which layers to be adapted')

    parser.add_argument('--batchadapt_bn_momentum', type=float, default=0.1,
                        help='batch adapt bn momemtum')

    parser.add_argument('--bn_stat_share_lr', type=str, default='all',
                        # choices=['all', 'block', 'layer', 'none'],
                        help='whether to print a lot')

    parser.add_argument('--grad_norm', type=str, default='sqrt_numel',
                        # choices=['none', 'sqrt_numel', 'numel'],
                        help='how to normalize gradient')

    parser.add_argument('--test', type=str, default='batch',
                        help='batch | online_avg')

    parser.add_argument('--load_adapt_path', type=str, default='none',
                        help='path to load adaptation rates')

    parser.add_argument('--load_adapt_idx', type=int, default=0,
                        help='rank in the pickle file')

    parser.add_argument('--load_adapt_round', type=int, default=-1,
                        help='which round to load')

    parser.add_argument('--surgical_metric', type=str, default='valid',
                        help='mode of surgical')

    parser.add_argument('--conf_mode', type=str, default='hard',
                        help='mode of constructing confusion matrix')

    parser.add_argument('--shot_beta', type=float, default=0,
                        help='beta used for shot')

    parser.add_argument('--em_epochs', type=int, default=2,
                        help='beta used for shot')

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Other Config
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    # training
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train ')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers of dataloader')

    # directories
    parser.add_argument('--data_dir', type=str, default='~/data',
                        help='where the data is stored')

    parser.add_argument('--partition_dir', type=str, default='~/data/atp/partition',
                        help='where the data partition is stored')

    parser.add_argument('--history_path', type=str, default='none')

    parser.add_argument('--load_model_path', type=str, default='none')

    parser.add_argument('--save_model_path', type=str, default='none')

    # for debug
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to print a lot')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether to visualize ')

    args = parser.parse_args()

    # number of clients
    args.num_train_clients = round((1 - args.client_holdout) * args.num_clients)
    args.num_test_clients = args.num_clients - args.num_train_clients

    # in and out-dimension of model
    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)  # binary classification has one output

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)

    # the path of partition config
    if args.partition_seed is None:
        args.partition_seed = args.seed

    partition_filename = 'client_%d_partition_%s_seed_%d.pkl' % (
        args.num_clients, args.partition, args.partition_seed)
    args.partition_path = os.path.join(args.partition_dir, args.dataset, partition_filename)

    # the path for corrputed dataset
    corruption_filename = 'client_%d_partition_%s_corruption_%s_seed_%d.pkl' % (
        args.num_clients, args.partition, args.corruption, args.partition_seed)
    args.corruption_path = os.path.join(args.data_dir, 'ttp', args.dataset, corruption_filename)

    # the path of domain dataset
    if args.dataset == 'pacs_aug':
        args.domain_path = os.path.join(args.data_dir, 'ttp',
                                        args.dataset + '_seed_' + str(args.partition_seed) + '.pkl')

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args
