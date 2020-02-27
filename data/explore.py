# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/17
# __file__ = explore
# __desc__ =
import cv2
import numpy as np
from pathlib import Path
# p = r"F:\Resources\dataset\SegNet-Tutorial-master\CamVid\trainannot"
# for image in Path(p).iterdir():
#     img = cv2.imread(str(image))
#     # s = (img == 255).astype('int').sum()
#     s = np.max(img)
#     print(s)

def IOU(pred: np.ndarray, label, c):
    ptrue = (pred == c).astype(int)
    ltrue = (label == c).astype(int)
    TP = (ptrue == ltrue).sum()
    FP = (ltrue[ptrue] != 1).sum()
    FN = (ltrue[1 - ptrue] == 1).sum()
    return TP / (TP + FP + FN)

def MIOU(pred: np.ndarray, label, numClass):
    r = []
    for i in range(numClass):
        r.append(IOU(pred, label, i))
    return np.mean(r)

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

label = np.array([2,3,0,1,4,2,1])
pred = np.array([2,3,0,2,3,1,1])
print(intersectionAndUnion(pred,label,5))
print(MIOU(pred,label,5))