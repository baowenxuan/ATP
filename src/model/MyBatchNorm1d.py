import torch
import torch.nn as nn


class MyBatchNorm1d(nn.Module):

    def __init__(self, bn):
        super(MyBatchNorm1d, self).__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps

        self.num_batches_tracked = bn.num_batches_tracked

        self.running_mean = nn.parameter.Parameter(bn.running_mean.detach().clone(), True)
        self.running_var = nn.parameter.Parameter(bn.running_var.detach().clone(), True)

        self.weight = nn.parameter.Parameter(bn.weight.detach().clone())
        self.bias = nn.parameter.Parameter(bn.bias.detach().clone())

        self.snapshot_mean = None
        self.snapshot_var = None

        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def forward(self, x):
        # save feature statistics for backward
        with torch.no_grad():
            x_t = x.data.permute((1, 0)).reshape((self.num_features, -1)).detach().clone()
            self.snapshot_mean = x_t.mean(dim=1)
            self.snapshot_var = x_t.var(dim=1)
            self.num_batches_tracked += 1

        if not self.training:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        else:
            var_forward = x_t.var(dim=1, unbiased=False)
            x = (x - self.snapshot_mean) / torch.sqrt(var_forward + self.eps)

        x = x * self.weight + self.bias

        return x

    def set_running_stat_grads(self):
        with torch.no_grad():
            self.running_mean.grad = self.running_mean.data - self.snapshot_mean
            self.running_var.grad = self.running_var.data - self.snapshot_var
            # self.running_var.grad = torch.zeros_like(self.running_var.grad)

    def clip_running_var(self):
        with torch.no_grad():
            self.running_var.clamp_(min=0)


class ModifiedBatchNorm1d(nn.Module):
    """
    BN with modified forward pass, similar to
    https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
    """

    def __init__(self, bn, prior):
        super(ModifiedBatchNorm1d, self).__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps

        self.num_batches_tracked = bn.num_batches_tracked

        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.weight = bn.weight
        self.bias = bn.bias

        self.prior = prior

        self.training = bn.training

    def forward(self, input):
        est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
        est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
        nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
        running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
        running_var = self.prior * self.running_var + (1 - self.prior) * est_var
        return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)
