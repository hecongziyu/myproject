# -*- coding: UTF-8 -*-
'''
https://pymupdf.readthedocs.io/en/latest/
'''
import fitz
import numpy as np
import random
import cv2
# import lib.im2latex.gen_latex_img as lxu
import os
from pylatex.base_classes import Environment, CommandBase, Arguments
from pylatex.package import Package
from pylatex import Document, Section, UnsafeCommand
from pylatex.utils import NoEscape
from matplotlib import pyplot as plt

def get_pdf_number(pdf_file):
    page_count = 0
    doc = None
    try:
        with open(pdf_file, 'rb') as f:
            data = f.read()
        doc = fitz.open('pdf', data)
        page_count = doc.pageCount
    finally:
        if doc is not None:
            doc.close()

    return page_count


#  block_flag 将文本区按块进行分割
def pdf_convert_image(pdf_file, page_number=0, page_height=None, block_flag=False):
    with open(pdf_file, 'rb') as f:
        pdf_datas=f.read()
    doc = fitz.open('pdf', pdf_datas)
    page = doc[page_number]
    pix = page.getPixmap()
    zoom_x = 1
    if page_height is not None:
        zoom_x = page_height/pix.height
        zoom_y = page_height/pix.height
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.getPixmap(matrix=mat,alpha=False)
    # text = page.getText().encode("utf8")
    words = []
    # 返回后三位： block_no, line_no, word_no
    image_data = pix.getImageData()
    page_words = page.getTextWords()
    if block_flag:
        words = [[int(x[0] * zoom_x),int(x[1] * zoom_x),
                  int(x[2] * zoom_x),int(x[3] * zoom_x),x[4],x[5],x[6],x[7]] for x in page_words]
        words_array = np.array(words)
        words_block = []
        block_uniqe_no = np.unique(words_array[:,5]).astype(np.int)
        block_uniqe_no.sort()
        block_uniqe_no = block_uniqe_no.astype(np.str)

        # print('blocks uniqe no:', block_uniqe_no, type(block_uniqe_no))
        for idx in block_uniqe_no:
            blocks = np.array(words_array[np.where(words_array[:,5]==idx),:])[0]
            # print('{} blocks: {}'.format(idx, blocks))
            line_uniqe_no = np.unique(blocks[:,1])
            for lidx in line_uniqe_no:
                lines = blocks[np.where(blocks[:,1]==lidx),:][0]
                x0 = np.min([int(x) for x in lines[:,0]])
                y0 = np.min([int(x) for x in lines[:,1]])
                x1 = np.max([int(x) for x in lines[:,2]])
                y1 = np.max([int(x) for x in lines[:,3]])
                l_words = ''.join(lines[:,4])
                words_block.append([x0,y0,x1,y1, l_words, idx, lidx])
        return image_data, words_block
    else:
        words = [[int(x[0] * zoom_x),int(x[1] * zoom_x),
                  int(x[2] * zoom_x),int(x[3] * zoom_x),x[4],x[5],x[6],x[7]] for x in page_words]     
        # words = [x for x in page_words]   
        return image_data, words



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="pdf parser")
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\pdf', type=str, help='path of the evaluated model')
    parser.add_argument('--file_name',default='2.pdf', type=str, help='path of the evaluated model')
    parser.add_argument('--page_number',default=6, type=int, help='path of the evaluated model')

    args = parser.parse_args()

    image, words = pdf_convert_image(os.path.sep.join([args.data_root, args.file_name]),page_number=args.page_number, block_flag=True)

    # print('words:', np.array([x[4] for x in words]))

    # np.savetxt(os.path.sep.join([args.data_root, '{}.txt'.format(args.file_name.split('.')[0])]),words)
    lines = '\n'.join([x[4] for x in words])     
    with open(os.path.sep.join([args.data_root, '{}.txt'.format(args.file_name.split('.')[0])]), 'w', encoding='utf-8') as f:
        f.write(lines)