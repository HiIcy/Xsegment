# coding:utf-8
# __author__ = hiicy redldw
# __time__ = 2020/1/14
# __file__ = Camvid
# __desc__ =
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Normalize,ToTensor


class CamvidSet(Dataset):
    def __init__(self,imgpath,annopath,transforms,numclass=32,img_size=(400,480)):
        self.imgpath = imgpath
        self.annopath = annopath
        self.images = os.listdir(self.imgpath)
        self.annos = os.listdir(self.annopath)
        self.transforms = transforms
        self.numclass = numclass
        self.height = img_size[0]
        self.width = img_size[1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        annofile = self.annos[item]
        img = os.path.join(self.imgpath,image)
        anno = os.path.join(self.annopath,annofile)
        im:Image.Image = Image.open(img)
        im = im.resize((self.width,self.height))
        ann = Image.open(anno)
        ann = ann.convert("L")
        ann = ann.resize((self.width,self.height))

        if self.transforms:
            for tranform in self.transforms:

                if isinstance(tranform,Normalize):
                    im = tranform(im)
                else:
                    im = tranform(im)
                    ann = tranform(ann)
        return im, ann
