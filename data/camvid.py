# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = Camvid
# __desc__ =
import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image,ImageFilter
from torchvision.transforms import Normalize,ToTensor


class CamvidSet(Dataset):
    def __init__(self,imgpath,annopath,transforms,numclass=32,return_name=False):
        self.imgpath = imgpath
        self.annopath = annopath
        self.images = os.listdir(self.imgpath)
        self.annos = os.listdir(self.annopath)
        self.transforms = transforms
        self.numclass = numclass
        self.return_name = return_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        annofile = self.annos[item]
        img = os.path.join(self.imgpath, image)
        anno = os.path.join(self.annopath, annofile)
        im = cv2.imread(img,cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = np.float32(im)
        ann = cv2.imread(anno,cv2.IMREAD_GRAYSCALE)
        assert im.shape[0]==ann.shape[0],"Image & label shape must match"
        assert im.shape[1]==ann.shape[1],"Image & label shape must match"

        if self.transforms:
            im,ann = self.transforms(im,ann)
        if self.return_name:
            return im,ann,image
        return im,ann
