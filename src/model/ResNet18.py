import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class ResNet18(Model):
    """
    ResNet
    """

    def __init__(self, shape_out):
        super(ResNet18, self).__init__()

        self.backbone = resnet18(pretrained=True)

        # replace the final fully connected layer
        del self.backbone.fc
        self.backbone.fc = nn.Linear(512, shape_out)

        self.drop_last = True

    def forward(self, x):
        if x.shape[1] == 1:  # convert 1-channel image to 3 channel
            x = x.repeat(1, 3, 1, 1)

        return self.backbone(x)

    def change_bn(self, mode='grad', prior=0):
        model = self.backbone

        if mode == 'grad':

            model.bn1 = MyBatchNorm2d(model.bn1)

            model.layer1[0].bn1 = MyBatchNorm2d(model.layer1[0].bn1)
            model.layer1[0].bn2 = MyBatchNorm2d(model.layer1[0].bn2)
            model.layer1[1].bn1 = MyBatchNorm2d(model.layer1[1].bn1)
            model.layer1[1].bn2 = MyBatchNorm2d(model.layer1[1].bn2)

            model.layer2[0].bn1 = MyBatchNorm2d(model.layer2[0].bn1)
            model.layer2[0].bn2 = MyBatchNorm2d(model.layer2[0].bn2)
            model.layer2[0].downsample[1] = MyBatchNorm2d(model.layer2[0].downsample[1])
            model.layer2[1].bn1 = MyBatchNorm2d(model.layer2[1].bn1)
            model.layer2[1].bn2 = MyBatchNorm2d(model.layer2[1].bn2)

            model.layer3[0].bn1 = MyBatchNorm2d(model.layer3[0].bn1)
            model.layer3[0].bn2 = MyBatchNorm2d(model.layer3[0].bn2)
            model.layer3[0].downsample[1] = MyBatchNorm2d(model.layer3[0].downsample[1])
            model.layer3[1].bn1 = MyBatchNorm2d(model.layer3[1].bn1)
            model.layer3[1].bn2 = MyBatchNorm2d(model.layer3[1].bn2)

            model.layer4[0].bn1 = MyBatchNorm2d(model.layer4[0].bn1)
            model.layer4[0].bn2 = MyBatchNorm2d(model.layer4[0].bn2)
            model.layer4[0].downsample[1] = MyBatchNorm2d(model.layer4[0].downsample[1])
            model.layer4[1].bn1 = MyBatchNorm2d(model.layer4[1].bn1)
            model.layer4[1].bn2 = MyBatchNorm2d(model.layer4[1].bn2)

        elif mode == 'prior':

            model.bn1 = ModifiedBatchNorm2d(model.bn1, prior=prior)

            model.layer1[0].bn1 = ModifiedBatchNorm2d(model.layer1[0].bn1, prior=prior)
            model.layer1[0].bn2 = ModifiedBatchNorm2d(model.layer1[0].bn2, prior=prior)
            model.layer1[1].bn1 = ModifiedBatchNorm2d(model.layer1[1].bn1, prior=prior)
            model.layer1[1].bn2 = ModifiedBatchNorm2d(model.layer1[1].bn2, prior=prior)

            model.layer2[0].bn1 = ModifiedBatchNorm2d(model.layer2[0].bn1, prior=prior)
            model.layer2[0].bn2 = ModifiedBatchNorm2d(model.layer2[0].bn2, prior=prior)
            model.layer2[0].downsample[1] = ModifiedBatchNorm2d(model.layer2[0].downsample[1], prior=prior)
            model.layer2[1].bn1 = ModifiedBatchNorm2d(model.layer2[1].bn1, prior=prior)
            model.layer2[1].bn2 = ModifiedBatchNorm2d(model.layer2[1].bn2, prior=prior)

            model.layer3[0].bn1 = ModifiedBatchNorm2d(model.layer3[0].bn1, prior=prior)
            model.layer3[0].bn2 = ModifiedBatchNorm2d(model.layer3[0].bn2, prior=prior)
            model.layer3[0].downsample[1] = ModifiedBatchNorm2d(model.layer3[0].downsample[1], prior=prior)
            model.layer3[1].bn1 = ModifiedBatchNorm2d(model.layer3[1].bn1, prior=prior)
            model.layer3[1].bn2 = ModifiedBatchNorm2d(model.layer3[1].bn2, prior=prior)

            model.layer4[0].bn1 = ModifiedBatchNorm2d(model.layer4[0].bn1, prior=prior)
            model.layer4[0].bn2 = ModifiedBatchNorm2d(model.layer4[0].bn2, prior=prior)
            model.layer4[0].downsample[1] = ModifiedBatchNorm2d(model.layer4[0].downsample[1], prior=prior)
            model.layer4[1].bn1 = ModifiedBatchNorm2d(model.layer4[1].bn1, prior=prior)
            model.layer4[1].bn2 = ModifiedBatchNorm2d(model.layer4[1].bn2, prior=prior)

    def set_running_stat_grads(self):

        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

    def freeze_bn_stats(self):
        """
        Do not update running stats of batch norm layers.
        """
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def set_layers_to_adapt(self, mode='all'):
        print(mode)
        self.backbone.train()
        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            # disable grad, to (re-)enable later
            self.backbone.requires_grad_(False)
            # configure norm for tent updates: enable grad + force batch statisics
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if mode in ['bn_all', 'bn_params']:
                        m.requires_grad_(True)
                    if mode in ['bn_all', 'bn_stat']:
                        m.track_running_stats = True
                        m.momentum = 1.0
                    else:
                        m.track_running_stats = False

                    # 04/04: we currently do not delete the running mean and var
                    # this might be updated if we find that updating them is beneficial.
                    # m.running_mean = None
                    # m.running_var = None

        elif mode == 'tent':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        elif mode == 'last_layer':
            # disable grad, to (re-)enable later
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    m.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'last_bias':
            # disable grad, to (re-)enable later
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    m.bias.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False  #

        elif mode == 'first_conv':
            # disable grad, to (re-)enable later
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            self.backbone.conv1.requires_grad_(True)

        elif mode == 'block1':
            self.backbone.requires_grad_(False)
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.requires_grad_(False)
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.requires_grad_(False)
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.requires_grad_(False)
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True


        elif mode == 'all':
            pass

        else:
            raise NotImplementedError

    def surgical(self, mode='all'):

        self.backbone.requires_grad_(False)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        if mode == 'block1':
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'last_layer':
            self.backbone.fc.requires_grad_(True)

        else:
            raise NotImplementedError

    def get_classifier(self):
        return self.backbone.fc

    def get_featurizer(self):
        resnet = self.backbone
        model = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', resnet.layer3),
            ('layer4', resnet.layer4),
            ('avgpool', resnet.avgpool),
            ('flatten', nn.Flatten()),
        ]))

        return model


def test():
    model = ResNet18(shape_out=10)
    model.change_bn()
    total_num = sum(p.numel() for name, p in model.state_dict().items())
    print(total_num)
    print(dict(model.named_parameters()).keys())
    print(len(dict(model.named_parameters())))

    # for m in list(model.modules()):
    #     print('-' * 100)
    #     print(m)
