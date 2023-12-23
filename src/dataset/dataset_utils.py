shapes_in = {
    'mnist': (1, 28, 28),
    'fmnist': (1, 28, 28),
    'cifar10': (3, 32, 32),
    'cifar10c': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'coarse-cifar100': (3, 32, 32),
    'digit': (3, 28, 28),
    'pacs': (3, 224, 224),
    'pacs_aug': (3, 224, 224),
    'vlcs': (3, 224, 224),
}

shapes_out = {
    'mnist': 10,
    'fmnist': 10,
    'cifar10': 10,
    'cifar10c': 10,
    'cifar100': 100,
    'coarse-cifar100': 20,
    'digit': 10,
    'pacs': 7,
    'pacs_aug': 7,
    'vlcs': 5,
}
