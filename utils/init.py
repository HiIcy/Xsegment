# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/15
# __file__ = init
# __desc__ =

import torch.nn as nn

def init_weight(*nets):
    for net in nets:
        for m in net.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

