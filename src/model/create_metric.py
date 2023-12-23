import torch


def create_metric(name='acc'):
    """
    metric function can be any function with scalar output.
    """
    if name == 'acc':
        return lambda logits, target: logits.argmax(dim=1).eq(target).float().mean()
    else:
        raise NotImplementedError('Unknown metric name: %s' % name)
    