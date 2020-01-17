# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = dataaugment
# __desc__ =
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic)

# transforms.RandomVerticalFlip