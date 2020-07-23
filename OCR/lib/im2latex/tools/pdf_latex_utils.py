# -*- coding: UTF-8 -*-
'''
https://pymupdf.readthedocs.io/en/latest/
https://wenku.baidu.com/view/63f270ac2379168884868762caaedd3383c4b5b1.html 
https://zhuanlan.zhihu.com/p/138116617
https://wenku.baidu.com/view/9f1832ac1fd9ad51f01dc281e53a580216fc50a9.html
http://www.texdoc.net/texmf-dist/doc/latex/tcolorbox/tcolorbox.pdf !
https://tex.stackexchange.com/questions/20575/attractive-boxed-equations ！
'''
import fitz
import numpy as np
import random
import cv2
import os
from pylatex.base_classes import Environment, CommandBase, Arguments
from pylatex.package import Package
from pylatex import Document, Section, UnsafeCommand
from pylatex.utils import NoEscape
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')




def gen_latex_pdf(data_root, file_name, formulas, latex_box_color='red'):
    # print('准备数据')
    # doc_text_color = [r'$$ \colorbox{red} {$ %s $}  $$' % x for x in formulas]
    doc_text = [r'\begin{empheq}[box={\mymath[colback=white,sharp corners]}]{equation*} %s \end{empheq}' % x for x in formulas]
    doc_text_color = [r'\begin{empheq}[box={\mymath[colback=red,sharp corners]}]{equation*} %s \end{empheq}' % x for x in formulas]

    gen_pdf(data_root, file_name, doc_text)
    gen_pdf(data_root, file_name + '_color', doc_text_color)




def gen_pdf(data_root, file_name, texts):
    
    # print('texts:', texts)
    file_path = os.path.sep.join([data_root,'pdf', file_name])
    # print('开始生成PDF文件, ', file_path)

    if os.path.exists(file_path):
        os.remove('{}.pdf'.format(file_path))

    doc = Document()
    doc.packages.add(Package('ctex'))
    doc.packages.add(Package('color'))
    doc.packages.add(Package('empheq'))

    # 数学符号字体
    doc.packages.add(Package('amsmath'))
    doc.packages.add(Package('xcolor'))  
    doc.packages.add(Package('graphicx'))  
    doc.packages.add(Package('geometry'))  
    doc.packages.add(Package('tcolorbox','most'))

    # doc.append(NoEscape(r'\usepackage[most]{tcolorbox}'))
    doc.append(NoEscape(r'\newgeometry{left=3cm,bottom=1cm}'))
    doc.append(NoEscape(r'\newtcbox{\mymath}[1][]{nobeforeafter, math upper, tcbox raise base,enhanced, boxrule=0pt,#1}'))

    for t in texts:
        # doc.append(NoEscape(r'~\\'))
        doc.append(NoEscape(t))

    doc.generate_pdf(file_path,clean_tex=True,compiler='xelatex')




# 取得PDF中带底纹latex文本位位置，并将不带底纹文本位置返回
# file_name 正常文件, color_file_name 带底纹文件
def gen_latex_img_pos(data_root, file_name,imgH=1200,image_dir='images',anno_dir='annotations',sub_dir='autogen',latexs_box_color='red'):
    pdf_file_path = os.path.sep.join([data_root,'pdf',file_name])

    anno_dir = os.path.sep.join([data_root,'data',anno_dir, sub_dir])
    if not os.path.exists(anno_dir):
        os.mkdir(anno_dir)


    # 得到数学公式坐标位置
    with open(f'{pdf_file_path}_color.pdf','rb') as f:
        data_color = f.read()

    image = pdf2image(data_color, imgH=imgH)
    image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)

    # cv2.imwrite('d:\\a.png', image)

    # plt.imshow(image)
    # plt.show()
    boundaries = [[0, 0, 255], [0, 0, 255]] # 红色
    lower = np.array(boundaries[0], dtype="uint8")
    upper = np.array(boundaries[1], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray,'gray')
    # plt.show()   
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # plt.imshow(binary,'gray')
    # plt.show()    

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    formula_pos = []

    # tmpImg = image.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        formula_pos.append([x,y,x+w,y+h,0])



    with open(f'{pdf_file_path}.pdf','rb') as f:
        data = f.read()
    image = pdf2image(data, imgH=imgH)
    image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
    img_dir = os.path.sep.join([data_root,'data', image_dir, sub_dir])
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    math_images = []

    # 根据Y轴进行排序
    formula_pos.sort(key=lambda x: x[1])

    # print('formula_pos:', formula_pos)



    for idx, pos in enumerate(formula_pos):    
        x0,y0,x1,y1,_ = pos
        f_image = image[y0:y1, x0:x1]
        # f_image = cv2.
        math_images.append(f_image)
        # cv2.imwrite(os.path.sep.join([img_dir, f'{file_name}_{idx}.png']), f_image)
    return math_images
    

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

