import torch.optim as optim


def create_optimizer(model, optimizer_name, lr):

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Unknown optimizer. ')

    return optimizer
