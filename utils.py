# -*- coding: utf-8 -*-
# @Time    : 2018/10/17 0017 9:06
# @Author  : Elliott Zheng
# @Email   : admin@hypercube.top
# @FileName: utils.py
import os

def mkdirs(dir):
    dir=os.path.abspath(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)