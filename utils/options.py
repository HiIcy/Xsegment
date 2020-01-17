# coding:utf-8
# __author__ = deepSea
# __time__ = 2020/1/15
# __file__ = options
# __desc__ =

import argparse
from .parser import parser_name

def parse_arg():
    parser = argparse.ArgumentParser(description="xsegmention net framework")
    # TODO:子解析器
    # subparsers = parser.add_subparsers(help="net framework")
    # for name in parser_name:
    #     subparsers.add_parser(name,parents=eval(f"get_{name}_parser"))

    parser.add_argument("--net",help="used net",type=str,default="pspnet")
    parser.add_argument("--batch",help="used batch size",type=int,default=16)
    parser.add_argument("--lr",help="used learning rate",type=float,default=0.01)
    parser.add_argument("--power",help="used learning rate",type=float,default=0.9)
    parser.add_argument("--momentum",help="used momentum",type=float,default=0.9)
    parser.add_argument("--decay",help="used weight decay",type=float,default=0.0001,dest="weight_decay")
    parser.add_argument("--iter",help="used iter times",type=int,default=1000)
    parser.add_argument("--mode",help="train or eval or inference",default="train")
    parser.add_argument("--config",help="use to help train",type=str)
    parser.add_argument("--pool_core",nargs="+",type=int,help="only pspnet need!")
    parser.add_argument("--aux_weight",type=float,help="aux loss weight")

    return parser.parse_args()


