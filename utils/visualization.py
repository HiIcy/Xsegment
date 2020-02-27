# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/18
# __file__ = visualization
# __desc__ =

import random
from PIL import Image
import numpy as np
color = [[random.randint(0,255)
                for i in range(3)]
                    for _ in range(100)]

colorf = r"/data/soft/javad/Xsegment/data/camvid/camvid_color.txt"
# with open(colorf,'w') as f:
#     for i in range(12):
#         f.write(','.join(map(lambda x:str(x),color[i]))+"\n")
colorg = []
with open(colorf,'r') as f:
    for line in f:
        s = line.split(",")
        s = list(map(lambda x:int(x),s))
        colorg.append(s)

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def draw_color(pridx,nclass=12):
    # color_group = [(i,random.choice(color)) for i in range(nclass)]
    # color_group = dict(color_group)

    orimage = np.zeros((pridx.shape[0], pridx.shape[1], 3))
    for c in range(nclass):
        # plocation =
        orimage[:,:,0] += ((pridx == c)*colorg[c][0]).astype('uint8')
        orimage[:,:,1] += ((pridx == c)*colorg[c][1]).astype('uint8')
        orimage[:,:,2] += ((pridx == c)*colorg[c][2]).astype('uint8')
    orimage = orimage.astype(np.uint8)
    return orimage

# import cv2
# pred = np.array([
#     [2,2,2,2,3,5,1,1],
#     [2,2,2,3,3,5,5,1],
#     [2,2,1,4,5,1,5,1],
#     [0,0,0,5,5,5,5,1],
#     [0,0,0,5,5,5,1,1]
# ])
# orj = draw_color(pred,6)
# orj = cv2.resize(orj,(100,100))
# print(orj)
# # print(orj[3:,3:6])
# cv2.imshow('sf',orj)
# cv2.waitKey(0)