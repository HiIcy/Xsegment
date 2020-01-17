# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = pspnet
# __desc__ =
import torch
from torchvision.models import resnet101
import torch.nn.functional as F
import torch.nn as nn
from utils.init import init_weight


class PyramidPoolModule(nn.Module):
    def __init__(self, scales, in_dim, reduction_dim):
        super(PyramidPoolModule,self).__init__()
        pools = []
        for scale in scales:
            pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(in_dim, reduction_dim, 1),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.pools = nn.ModuleList(pools)

    def forward(self, input: torch.Tensor):
        input_size = input.size()
        r = [input]
        for x in self.pools:
            r.append(F.upsample(x(input), input_size[2:], mode="bilinear", align_corners=True))  # align_corners 对齐左上角
        return torch.cat(r, dim=1)


class pspnet(nn.Module):
    def __init__(self,numclass,pretrained=True,use_aux=True):
        super(pspnet,self).__init__()
        self.use_aux = use_aux
        self.numclass = numclass
        pth="/data/soft/javad/cache/resnet101-5d3b4d8f.pth"
        resnet = resnet101(False)
        if pretrained:
            resnet.load_state_dict(torch.load(pth))
        # 直接继承resnet的特征提取模块
        self.layer0 = nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 对layer3进行改进，空洞卷积
        for n,m in self.layer3.named_modules():
            if 'conv2' in n:
                # REW:通过扩张卷积,不影响输出的特征图尺寸，但是扩大了感受野
                m.dilation,m.padding,m.stride = (2,2),(2,2),(1,1)
            elif "downsample.0" in n:
                # 步长改为1 就不减小图片尺寸
                m.stride = (1,1)
        for n,m in self.layer4.named_modules():
            if 'conv2' in n:
                # 扩大了感受野
                m.dilation,m.padding,m.stride = (4,4),(4,4),(1,1)
            elif 'downsample.0' in n:
                m.stride = (1,1)
        # 2048 是残差的两个1024加起来
        self.ppm = PyramidPoolModule([1,2,3,6],2048,512)
        # FAQ:这里如何把图片恢复的
        self.fcn = nn.Sequential(
            # 特征图尺寸不变
            nn.Conv2d(4096,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(512,numclass,kernel_size=1)
        )
        if self.use_aux:
            self.aux_logits = nn.Conv2d(1024,numclass,kernel_size=1)
            init_weight(self.aux_logits)
        init_weight(self.ppm,self.fcn)

    def forward(self,input:torch.Tensor):
        x_size = input.size()
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        mas_x = self.layer4(x)
        mas_x = self.ppm(mas_x)
        mas_x = self.fcn(mas_x)
        if self.use_aux and self.training:
            aux_x = self.aux_logits(x)
            # FAQ:在这里恢复尺寸，能达到效果?
            # FIXME:F.upsample_bilinear() 待试用?
            return (F.upsample(mas_x,x_size[2:],mode="bilinear"),
                    F.upsample(aux_x,x_size[2:],mode="bilinear"))
        return F.upsample(mas_x,x_size[2:],mode="bilinear")
