# -*- coding: UTF-8 -*-
# https://github.com/IBM/pytorch-seq2seq/tree/master/seq2seq
import argparse
from model import *


def train(args):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test paper line classify')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\testpaper', type=str, help='path of the data')
    parser.add_argument('--max_epoch',default=100, type=int, help='path of the data')

    args = parser.parse_args()
    train(args)
