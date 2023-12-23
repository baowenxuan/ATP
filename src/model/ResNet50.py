import torch
import torch.nn as nn
from torchvision.models import resnet50
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class ResNet50(Model):
    """
    ResNet
    """

    def __init__(self, shape_out):
        super(ResNet50, self).__init__()

        self.backbone = resnet50(pretrained=True)

        # replace the final fully connected layer

        print(self.backbone.fc)
        del self.backbone.fc
        self.backbone.fc = nn.Linear(2048, shape_out)
        print(self.backbone.fc)

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
            model.layer1[0].bn3 = MyBatchNorm2d(model.layer1[0].bn3)
            model.layer1[0].downsample[1] = MyBatchNorm2d(model.layer1[0].downsample[1])
            model.layer1[1].bn1 = MyBatchNorm2d(model.layer1[1].bn1)
            model.layer1[1].bn2 = MyBatchNorm2d(model.layer1[1].bn2)
            model.layer1[1].bn3 = MyBatchNorm2d(model.layer1[1].bn3)
            model.layer1[2].bn1 = MyBatchNorm2d(model.layer1[2].bn1)
            model.layer1[2].bn2 = MyBatchNorm2d(model.layer1[2].bn2)
            model.layer1[2].bn3 = MyBatchNorm2d(model.layer1[2].bn3)

            model.layer2[0].bn1 = MyBatchNorm2d(model.layer2[0].bn1)
            model.layer2[0].bn2 = MyBatchNorm2d(model.layer2[0].bn2)
            model.layer2[0].bn3 = MyBatchNorm2d(model.layer2[0].bn3)
            model.layer2[0].downsample[1] = MyBatchNorm2d(model.layer2[0].downsample[1])
            model.layer2[1].bn1 = MyBatchNorm2d(model.layer2[1].bn1)
            model.layer2[1].bn2 = MyBatchNorm2d(model.layer2[1].bn2)
            model.layer2[1].bn3 = MyBatchNorm2d(model.layer2[1].bn3)
            model.layer2[2].bn1 = MyBatchNorm2d(model.layer2[2].bn1)
            model.layer2[2].bn2 = MyBatchNorm2d(model.layer2[2].bn2)
            model.layer2[2].bn3 = MyBatchNorm2d(model.layer2[2].bn3)
            model.layer2[3].bn1 = MyBatchNorm2d(model.layer2[3].bn1)
            model.layer2[3].bn2 = MyBatchNorm2d(model.layer2[3].bn2)
            model.layer2[3].bn3 = MyBatchNorm2d(model.layer2[3].bn3)

            model.layer3[0].bn1 = MyBatchNorm2d(model.layer3[0].bn1)
            model.layer3[0].bn2 = MyBatchNorm2d(model.layer3[0].bn2)
            model.layer3[0].bn3 = MyBatchNorm2d(model.layer3[0].bn3)
            model.layer3[0].downsample[1] = MyBatchNorm2d(model.layer3[0].downsample[1])
            model.layer3[1].bn1 = MyBatchNorm2d(model.layer3[1].bn1)
            model.layer3[1].bn2 = MyBatchNorm2d(model.layer3[1].bn2)
            model.layer3[1].bn3 = MyBatchNorm2d(model.layer3[1].bn3)
            model.layer3[2].bn1 = MyBatchNorm2d(model.layer3[2].bn1)
            model.layer3[2].bn2 = MyBatchNorm2d(model.layer3[2].bn2)
            model.layer3[2].bn3 = MyBatchNorm2d(model.layer3[2].bn3)
            model.layer3[3].bn1 = MyBatchNorm2d(model.layer3[3].bn1)
            model.layer3[3].bn2 = MyBatchNorm2d(model.layer3[3].bn2)
            model.layer3[3].bn3 = MyBatchNorm2d(model.layer3[3].bn3)
            model.layer3[4].bn1 = MyBatchNorm2d(model.layer3[4].bn1)
            model.layer3[4].bn2 = MyBatchNorm2d(model.layer3[4].bn2)
            model.layer3[4].bn3 = MyBatchNorm2d(model.layer3[4].bn3)
            model.layer3[5].bn1 = MyBatchNorm2d(model.layer3[5].bn1)
            model.layer3[5].bn2 = MyBatchNorm2d(model.layer3[5].bn2)
            model.layer3[5].bn3 = MyBatchNorm2d(model.layer3[5].bn3)

            model.layer4[0].bn1 = MyBatchNorm2d(model.layer4[0].bn1)
            model.layer4[0].bn2 = MyBatchNorm2d(model.layer4[0].bn2)
            model.layer4[0].bn3 = MyBatchNorm2d(model.layer4[0].bn3)
            model.layer4[0].downsample[1] = MyBatchNorm2d(model.layer4[0].downsample[1])
            model.layer4[1].bn1 = MyBatchNorm2d(model.layer4[1].bn1)
            model.layer4[1].bn2 = MyBatchNorm2d(model.layer4[1].bn2)
            model.layer4[1].bn3 = MyBatchNorm2d(model.layer4[1].bn3)
            model.layer4[2].bn1 = MyBatchNorm2d(model.layer4[2].bn1)
            model.layer4[2].bn2 = MyBatchNorm2d(model.layer4[2].bn2)
            model.layer4[2].bn3 = MyBatchNorm2d(model.layer4[2].bn3)

        elif mode == 'prior':

            model.layer1[0].bn1 = ModifiedBatchNorm2d(model.layer1[0].bn1, prior=prior)
            model.layer1[0].bn2 = ModifiedBatchNorm2d(model.layer1[0].bn2, prior=prior)
            model.layer1[0].bn3 = ModifiedBatchNorm2d(model.layer1[0].bn3, prior=prior)
            model.layer1[0].downsample[1] = ModifiedBatchNorm2d(model.layer1[0].downsample[1], prior=prior)
            model.layer1[1].bn1 = ModifiedBatchNorm2d(model.layer1[1].bn1, prior=prior)
            model.layer1[1].bn2 = ModifiedBatchNorm2d(model.layer1[1].bn2, prior=prior)
            model.layer1[1].bn3 = ModifiedBatchNorm2d(model.layer1[1].bn3, prior=prior)
            model.layer1[2].bn1 = ModifiedBatchNorm2d(model.layer1[2].bn1, prior=prior)
            model.layer1[2].bn2 = ModifiedBatchNorm2d(model.layer1[2].bn2, prior=prior)
            model.layer1[2].bn3 = ModifiedBatchNorm2d(model.layer1[2].bn3, prior=prior)

            model.layer2[0].bn1 = ModifiedBatchNorm2d(model.layer2[0].bn1, prior=prior)
            model.layer2[0].bn2 = ModifiedBatchNorm2d(model.layer2[0].bn2, prior=prior)
            model.layer2[0].bn3 = ModifiedBatchNorm2d(model.layer2[0].bn3, prior=prior)
            model.layer2[0].downsample[1] = ModifiedBatchNorm2d(model.layer2[0].downsample[1], prior=prior)
            model.layer2[1].bn1 = ModifiedBatchNorm2d(model.layer2[1].bn1, prior=prior)
            model.layer2[1].bn2 = ModifiedBatchNorm2d(model.layer2[1].bn2, prior=prior)
            model.layer2[1].bn3 = ModifiedBatchNorm2d(model.layer2[1].bn3, prior=prior)
            model.layer2[2].bn1 = ModifiedBatchNorm2d(model.layer2[2].bn1, prior=prior)
            model.layer2[2].bn2 = ModifiedBatchNorm2d(model.layer2[2].bn2, prior=prior)
            model.layer2[2].bn3 = ModifiedBatchNorm2d(model.layer2[2].bn3, prior=prior)
            model.layer2[3].bn1 = ModifiedBatchNorm2d(model.layer2[3].bn1, prior=prior)
            model.layer2[3].bn2 = ModifiedBatchNorm2d(model.layer2[3].bn2, prior=prior)
            model.layer2[3].bn3 = ModifiedBatchNorm2d(model.layer2[3].bn3, prior=prior)

            model.layer3[0].bn1 = ModifiedBatchNorm2d(model.layer3[0].bn1, prior=prior)
            model.layer3[0].bn2 = ModifiedBatchNorm2d(model.layer3[0].bn2, prior=prior)
            model.layer3[0].bn3 = ModifiedBatchNorm2d(model.layer3[0].bn3, prior=prior)
            model.layer3[0].downsample[1] = ModifiedBatchNorm2d(model.layer3[0].downsample[1], prior=prior)
            model.layer3[1].bn1 = ModifiedBatchNorm2d(model.layer3[1].bn1, prior=prior)
            model.layer3[1].bn2 = ModifiedBatchNorm2d(model.layer3[1].bn2, prior=prior)
            model.layer3[1].bn3 = ModifiedBatchNorm2d(model.layer3[1].bn3, prior=prior)
            model.layer3[2].bn1 = ModifiedBatchNorm2d(model.layer3[2].bn1, prior=prior)
            model.layer3[2].bn2 = ModifiedBatchNorm2d(model.layer3[2].bn2, prior=prior)
            model.layer3[2].bn3 = ModifiedBatchNorm2d(model.layer3[2].bn3, prior=prior)
            model.layer3[3].bn1 = ModifiedBatchNorm2d(model.layer3[3].bn1, prior=prior)
            model.layer3[3].bn2 = ModifiedBatchNorm2d(model.layer3[3].bn2, prior=prior)
            model.layer3[3].bn3 = ModifiedBatchNorm2d(model.layer3[3].bn3, prior=prior)
            model.layer3[4].bn1 = ModifiedBatchNorm2d(model.layer3[4].bn1, prior=prior)
            model.layer3[4].bn2 = ModifiedBatchNorm2d(model.layer3[4].bn2, prior=prior)
            model.layer3[4].bn3 = ModifiedBatchNorm2d(model.layer3[4].bn3, prior=prior)
            model.layer3[5].bn1 = ModifiedBatchNorm2d(model.layer3[5].bn1, prior=prior)
            model.layer3[5].bn2 = ModifiedBatchNorm2d(model.layer3[5].bn2, prior=prior)
            model.layer3[5].bn3 = ModifiedBatchNorm2d(model.layer3[5].bn3, prior=prior)

            model.layer4[0].bn1 = ModifiedBatchNorm2d(model.layer4[0].bn1, prior=prior)
            model.layer4[0].bn2 = ModifiedBatchNorm2d(model.layer4[0].bn2, prior=prior)
            model.layer4[0].bn3 = ModifiedBatchNorm2d(model.layer4[0].bn3, prior=prior)
            model.layer4[0].downsample[1] = ModifiedBatchNorm2d(model.layer4[0].downsample[1], prior=prior)
            model.layer4[1].bn1 = ModifiedBatchNorm2d(model.layer4[1].bn1, prior=prior)
            model.layer4[1].bn2 = ModifiedBatchNorm2d(model.layer4[1].bn2, prior=prior)
            model.layer4[1].bn3 = ModifiedBatchNorm2d(model.layer4[1].bn3, prior=prior)
            model.layer4[2].bn1 = ModifiedBatchNorm2d(model.layer4[2].bn1, prior=prior)
            model.layer4[2].bn2 = ModifiedBatchNorm2d(model.layer4[2].bn2, prior=prior)
            model.layer4[2].bn3 = ModifiedBatchNorm2d(model.layer4[2].bn3, prior=prior)


    def set_running_stat_grads(self):

        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

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
            ('flatten', nn.Flatten())
        ]))

        return model


def test():
    model = ResNet50(shape_out=10)
    model.change_bn()
    total_num = sum(p.numel() for name, p in model.state_dict().items())
    print(total_num)
    print(dict(model.named_parameters()).keys())
    print(len(dict(model.named_parameters())))

    data = torch.randn((20, 3, 32, 32))
    print(model(data).shape)

    for m in list(model.modules()):
        print('-' * 100)
        print(m)