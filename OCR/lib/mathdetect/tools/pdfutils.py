# -*- coding: UTF-8 -*-
'''
https://pymupdf.readthedocs.io/en/latest/
https://www.jianshu.com/p/2bef8b44f40a 中文排版
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
import matplotlib
matplotlib.use('TkAgg')

ANNO = 'annotations'
IMAGE = 'images'

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
def get_image_area(pdf_datas, page_number=0, page_height=None, begin_word='开始', end_word='结束'):
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
    # print('page words:', page_words)

    begin_word_pos = [[int(x[0] * zoom_x),int(x[1] * zoom_x),int(x[2] * zoom_x),int(x[3] * zoom_x)] for x in page_words if x[4] == begin_word][0]
    end_word_pos = [[int(x[0] * zoom_x),int(x[1] * zoom_x),int(x[2] * zoom_x),int(x[3] * zoom_x)] for x in page_words if x[4] == end_word]

    if len(end_word_pos) > 0:
        image_area = [begin_word_pos[3],end_word_pos[0][1]]    
    else:
        image_area = [begin_word_pos[3],-1]    

    # print('begin word pos :', begin_word_pos)    
    # print('end word pos :', end_word_pos)    
    
    return image_area




# 取得PDF中带底纹latex文本位位置，并将不带底纹文本位置返回
# file_name 正常文件, color_file_name 带底纹文件
def gen_latex_img_pos(data_root, imgH=1024,image_dir='images',anno_dir='annotations',sub_dir='autogen',latexs_box_color='red'):
    # pdf_file_path = os.path.sep.join([data_root,'pdf',file_name])

    # anno_dir = os.path.sep.join([data_root,'data',anno_dir, sub_dir])
    # if not os.path.exists(anno_dir):
    #     os.mkdir(anno_dir)


    # 得到数学公式坐标位置
    with open(os.path.sep.join([data_root,'tmp','tmp_color.pdf']),'rb') as f:
        data_color = f.read()


    # 得到图片的范围，区间“开始”， “结束”
    image_area = get_image_area(data_color,page_height=imgH)
    # print('image area:', image_area)
    y0,y1 = image_area

    image = pdf2image(data_color, imgH=imgH)
    image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
    if y1 == -1:
        y1 = image.shape[0]

    image = image[y0:y1, 0:image.shape[1]-50]


    boundaries = [[0, 0, 255], [0, 0, 255]] # 红色
    lower = np.array(boundaries[0], dtype="uint8")
    upper = np.array(boundaries[1], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray,'gray')
    # plt.show()

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    formula_pos = []

    # tmpImg = image.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(tmpImg, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # formula_pos append (x1,y1,x2,y2, math label)
        formula_pos.append([x,y,x+w,y+h,0])

    # print('formula_pos: ', formula_pos)

    # print(tmpImg.shape)
    # print(np.array(formula_pos))

    # plt.imshow(tmpImg)
    # plt.show()

    # np.savetxt(os.path.sep.join([anno_dir, f'{file_name}.pmath']),np.array(formula_pos),'%.3f', ',', )


    # 得到PDF中图片位置
    boundaries = [[255, 0, 0], [255, 0, 0]] # 蓝色
    lower = np.array(boundaries[0], dtype="uint8")
    upper = np.array(boundaries[1], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pic_pos = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
         # pic_pos append (x1,y1,x2,y2, pic label)
        pic_pos.append([x,y,x+w,y+h,1])

    # print('pic_pos: ', pic_pos)
    # np.savetxt(os.path.sep.join([anno_dir, f'{file_name}.ppic']),np.array(pic_pos),'%.3f', ',', )


    with open(os.path.sep.join([data_root,'tmp','tmp.pdf']),'rb') as f:
        data = f.read()
    image = pdf2image(data, imgH=imgH)
    image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
    image = image[y0:y1, 0:image.shape[1]-50]

    # for _pos in formula_pos:
    #     x0,y0,x1,y1,_ = _pos
    #     cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 2)

    # for _pos in pic_pos:
    #     x0,y0,x1,y1,_ = _pos
    #     cv2.rectangle(image, (x0,y0), (x1, y1), (0, 0, 255), 2)


    # plt.imshow(image)
    # plt.show()        

    return image,formula_pos, pic_pos


def pdf2image(pdf_datas, page_number=0, imgH=None):
    doc = fitz.open('pdf', pdf_datas)
    page = doc[page_number]
    pix = page.getPixmap()
    zoom_x = 1
    if imgH is not None:
        zoom_x = imgH/pix.height
        zoom_y = imgH/pix.height
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.getPixmap(matrix=mat,alpha=False)
    image_data = pix.getImageData()
    return image_data






