# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/15
# __file__ = metrics
# __desc__ =
from copy import deepcopy

import numpy as np


#######  单个的PA/IOU  #########
def PA(pred: np.ndarray, label):
    """
    :param pred:[H,W]
    :param label: [H,W]
    :return:
    """
    pH, pW = pred.size
    p_sum = pH * pW
    pl = pred - label
    equal_sum = (pl == 0).sum()
    return equal_sum / p_sum


def MPA(pred: np.ndarray, label, numClass):
    r = []
    for i in range(numClass):
        tlabel = deepcopy(label)
        tpred = deepcopy(pred)
        tp_sum = (tlabel == i).sum()
        tlocation = tlabel == i
        e_sum = (tpred[tlocation] == i).sum()
        r.append(e_sum / tp_sum)
    return np.mean(r)


# 语义分割的iou、两个集合的交,并之比
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


### 基于 confusionMatrix  ##########
class Metrics:
    def __init__(self, numclass=12):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass, self.numclass))

    def pixelAccuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        molecule = np.diag(self.confusionMatrix) #提取对角线 TP
        molecule = np.sum(molecule)
        denominator = np.sum(self.confusionMatrix)
        return molecule / (denominator+1e-6)

    def classPixelAccuracy(self):
        # pi = TP / TP + FP
        PIs = np.sum(self.confusionMatrix,axis=1)
        molecule = np.diag(self.confusionMatrix)
        return molecule / PIs

    def meanPixelAccuracy(self):
        classpixelacc = self.classPixelAccuracy()
        meanacc = np.nanmean(classpixelacc)
        return meanacc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        FP = np.sum(self.confusionMatrix,axis=1)
        FN = np.sum(self.confusionMatrix,axis=0)
        union = FP + FN - intersection
        oiou = intersection / union
        meaniou = np.nanmean(oiou)
        return meaniou

    def genConfusionMatrix(self, pred, label):
        # 背景也算上
        mask = ((0 <= label) & (label < self.numclass))
        # REW: 计算所有类的混淆矩阵
        board = self.numclass*label[mask]+pred[mask]
        board = board.astype("int")
        board = np.bincount(board,minlength=self.numclass**2)
        confusionMatrix = board.reshape((self.numclass,self.numclass))
        return confusionMatrix

    def addBatch(self,imgpred,imglabel):
        assert imgpred.shape == imglabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgpred,imglabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numclass, self.numclass))

    def loadData(self,bpred,blabel):
        assert bpred.shape == blabel.shape
        N = bpred.shape[0]
        for i in range(N):
            self.addBatch(bpred[i],blabel[i])

