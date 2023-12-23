import torch.nn as nn
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm1d import MyBatchNorm1d, ModifiedBatchNorm1d
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class ShallowCNN(Model):
    """
    Shallow CNN, minimum input width and height is 18.
    """
    def __init__(self, shape_in, shape_out):
        super(ShallowCNN, self).__init__()

        in_channels = shape_in[0]
        h = ((((shape_in[1] - 2) // 2) - 2) // 2) - 2
        w = ((((shape_in[2] - 2) // 2) - 2) // 2) - 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=h * w * 64, out_features=64)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(in_features=64, out_features=shape_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(x.shape[0], -1)  # (num_samples, num_features)
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.linear5(x)

        return x

    def change_bn(self, mode='grad', prior=0):

        if mode == 'grad':

            self.bn1 = MyBatchNorm2d(self.bn1)
            self.bn2 = MyBatchNorm2d(self.bn2)
            self.bn3 = MyBatchNorm2d(self.bn3)
            self.bn4 = MyBatchNorm1d(self.bn4)

        elif mode == 'prior':

            self.bn1 = ModifiedBatchNorm2d(self.bn1, prior=prior)
            self.bn2 = ModifiedBatchNorm2d(self.bn2, prior=prior)
            self.bn3 = ModifiedBatchNorm2d(self.bn3, prior=prior)
            self.bn4 = ModifiedBatchNorm1d(self.bn4, prior=prior)

    def set_running_stat_grads(self):

        for m in self.modules():
            if isinstance(m, MyBatchNorm2d) or isinstance(m, MyBatchNorm1d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.modules():
            if isinstance(m, MyBatchNorm2d) or isinstance(m, MyBatchNorm1d):
                m.clip_running_var()


    def surgical(self, mode='all'):

        self.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, MyBatchNorm2d) or isinstance(m, MyBatchNorm1d):
                m.track_running_stats = False

        if mode == 'block1':
            self.conv1.requires_grad_(True)
            self.bn1.requires_grad_(True)
            self.bn1.track_running_stats = True

        elif mode == 'block2':
            self.conv2.requires_grad_(True)
            self.bn2.requires_grad_(True)
            self.bn2.track_running_stats = True

        elif mode == 'block3':
            self.conv3.requires_grad_(True)
            self.bn3.requires_grad_(True)
            self.bn3.track_running_stats = True

        elif mode == 'block4':
            self.linear4.requires_grad_(True)
            self.bn4.requires_grad_(True)
            self.bn4.track_running_stats = True

        elif mode == 'last_layer':
            self.linear5.requires_grad_(True)

        else:
            raise NotImplementedError

    def get_classifier(self):
        return self.linear5

    def get_featurizer(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', self.conv1),
            ('bn1', self.bn1),
            ('relu1', self.relu1),
            ('pool1', self.pool1),
            ('conv2', self.conv2),
            ('bn2', self.bn2),
            ('relu2', self.relu2),
            ('pool2', self.pool2),
            ('conv3', self.conv3),
            ('bn3', self.bn3),
            ('relu3', self.relu3),
            ('flatten', nn.Flatten()),
            ('linear4', self.linear4),
            ('bn4', self.bn4),
            ('relu4', self.relu4),
        ]))

        return model



def test():
    import torch
    # x = torch.randn([1, 3, 2, 2])
    # conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
    # print(conv(x).shape)

    model = ShallowCNN(shape_in=(3, 18, 29), shape_out=10)
    x = torch.randn(2, 3, 18, 29)
    print(model(x).shape)
