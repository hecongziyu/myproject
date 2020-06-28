# -*- coding: UTF-8 -*-

import subprocess
import os
import time
import base64
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
from io import BytesIO

'''
https://www.jianshu.com/p/a6df4c177d62
调用libreoffice 转换word为html
D:\tools\libreoffice\program>soffice.bin --convert-to html d:\公司战略学习心得.docx --outdir d:\data
'''


def convert_docx(process, file_name, output_dir):
    start_time = time.time()
    print('file name:', file_name)
    print('process:', process)
    # subprocess.call([process, '--convert-to', 'html:XHTML Writer File:UTF8', file_name, '--outdir', output_dir])
    subprocess.call([process,'--headless' ,'--convert-to', 'html','--convert-images-to','jpg', file_name, '--outdir', output_dir])
    print('use time:' , (time.time() - start_time))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="pdf parser")
    parser.add_argument('--process',default='D:\\tools\\libreoffice\\program\\soffice', type=str, help='path of the libreoffice')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\testpaper\\source\\paper', type=str, help='data root path of origin test paper')
    parser.add_argument('--file_name',default=u'207雅礼高一上第三次月考-教师版.docx', type=str, help='path of the evaluated model')
    parser.add_argument('--output_dir',default='D:\\PROJECT_TW\\git\\data\\testpaper\\output', type=str, help='convert path')
    args = parser.parse_args()

    convert_docx(args.process, os.path.sep.join([args.data_root, args.file_name]), args.output_dir)
    # convert_base_64()
