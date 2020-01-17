# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = resnet
# __desc__ =

import torch
import torch.nn as nn
from torchvision.models import resnet101

__all__ = ["res_stage3"]


def BN_CONV(inplanes, planes, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(inplanes, planes, kernel_size, stride),
        nn.BatchNorm2d(planes)
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes:input channels
        :param planes: output channels
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()
        self.conv_bn_1 = BN_CONV(inplanes, planes, 1)
        self.conv_bn_2 = BN_CONV(planes, planes, 3, stride)
        self.conv_bn_3 = BN_CONV(planes, planes * self.expansion, 1, stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        x = self.conv_bn_1(x)
        x = self.relu(x)

        x = self.conv_bn_2(x)
        x = self.relu(x)

        x = self.conv_bn_3(x)

        if self.downsample:
            # 对原输入做一个变换
            identify = self.downsample(identify)
        x += identify
        x = self.relu(x)

        return x


class ret(nn.Module):
    def __init__(self, block, layers, num_classes=3, zero_init_residual=False):
        super(ret, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # TODO:


res = resnet101(pretrained=True)
import torchsummary
torchsummary.summary(res,(3,400,400),1,"cpu")
# for n,m in res.layer4.named_modules():
#     print(n,' -> ',m)
res_stage3 = nn.Sequential(*list(res.children())[:6])

# t = torch.rand(2,3,400,400)
# print(res_stage3)