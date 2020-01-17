# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/15
# __file__ = train
# __desc__ =
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import warnings
warnings.filterwarnings("ignore")
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from utils.options import parse_arg
from torch.optim import SGD
import torch.nn as nn
import torch

if __name__ == "__main__":
    #torch.multiprocessing.freeze_support()

    from operate.reader import Processer
    args = parse_arg()
    crition = nn.CrossEntropyLoss()
    optimzer = SGD

    processer = Processer(args,crition,optimzer)

    processer.train()

    print("train done!")
