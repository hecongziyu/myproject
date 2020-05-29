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
def pdf_convert_image(pdf_datas, page_number=0, page_height=None, block_flag=False):
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
		block_uniqe_no = np.unique(words_array[:,5])

		for idx in block_uniqe_no:
			blocks = np.array(words_array[np.where(words_array[:,5]==idx),:])[0]
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
		return image_data, page_words


# 以图片方式随机替换PDF内容中的文字
'''
 image, insert_image  : opencv2 image
 words block : [x0,y0,x1,y1, words, block_no, line_no]
 返回 以替换的图片，及替换内容的坐标位置 
 单行公式高度: 330, 带根号公式高度： 435,  多行公式高度: 560, 注意后期需取大量数据做比较
'''
def replace_pdf_content(words, insert_image):
    words_heigh = words[3] - words[1]
    words_width = words[2] - words[0]
    radio = (words_heigh / insert_image.shape[0]) * (insert_image.shape[0]/330)
    dim = (int(insert_image.shape[1] * radio), int(insert_image.shape[0] * radio))
    t_image = cv2.resize(insert_image, dim, interpolation=cv2.INTER_AREA)
    x_position = words[0] + random.randint(0, words_width)
    y_position = words[1] - int((dim[1] - words_heigh)/2 )
    return (x_position, y_position,x_position + dim[0], y_position + dim[1]), t_image


def batch_gen_image_pdf(image, words_block, latex_image_lists, size=1):
    latex_pos_lists = []
    for idx in range(size):
        words = words_block[random.randint(0, len(words_block)-1)]
        # 需另外处理，处理已更新的位置，需过滤掉, 如果在同一行，需进行额外处理
        pos, tmg = replace_pdf_content(words, latex_image_lists[random.randint(0,len(latex_image_lists)-1)])
        if not __is_in_pos__(latex_pos_lists, pos):
            image[pos[1] : pos[3], pos[0]:pos[2]] = tmg
            latex_pos_lists.append(pos)		
    return image, latex_pos_lists

def __is_in_pos__(pos_lists, pos):
    flag = False
    x0,y0,x1,y1 = pos[0:4]
    for pitem in pos_lists:
        if (y0 >= pitem[1] and y0 <= pitem[3]) and (x0 >= pitem[0] and x0 <= pitem[2]):
            flag = True
            break
    return flag


# 生成数学公式检测训练数据, 读取PDF文件转换成图片，将数学公式图片替换到图片中
def gen_test_pos_data(pdf_file,page_number,save_path,ext_dir_name, dir_name='gen'):
    with open(pdf_file, 'rb') as f:
        data=f.read()
        pimage, ptext = pdf_convert_image(pdf_datas=data,page_number=page_number,page_height=1024,block_flag=True)
    image = cv2.imdecode(np.frombuffer(pimage, np.uint8),cv2.IMREAD_COLOR)
    latex_lists = lxu.random_latex(20)
    limage_lists = [lxu.get_latex_image(x) for x in  latex_lists]
    image, pos = batch_gen_image_pdf(image,ptext,limage_lists,size=random.randint(5,15))

    image_dir = os.path.sep.join([save_path, IMAGE, dir_name, ext_dir_name])
    if not os.path.exists(image_dir):
    	os.mkdir(image_dir)
    cv2.imwrite(os.path.sep.join([image_dir, f'{page_number}.png']), image)
    # print(type(pos))
    # print(pos)

    anno_dir = os.path.sep.join([save_path, ANNO, dir_name, ext_dir_name])
    if not os.path.exists(anno_dir):
    	os.mkdir(anno_dir)
    np.savetxt(os.path.sep.join([anno_dir, f'{page_number}.pmath']),np.array(pos),'%.3f', ',', )



# 生成带latex文本的PDF文件，返回两个PDF， 一个带颜色底纹，一个不带
# texts 正常文本 List， latexs 需增加的latexs
def gen_latex_pdf(data_root, file_name, texts, latexs,max_phrase=10,latex_box_color='red'):
    
    # 随机插入latexs, 分为带颜色和不带颜色的
    # 随机插入几何图形， 边框颜色为蓝色

    # print(os.path.sep.join([data_root,'picture']))
    pic_files = os.listdir(os.path.sep.join([data_root,'picture']))
    pic_files = [os.path.sep.join([data_root, 'picture',x]).replace('\\','/') for x in pic_files]



    doc_text = []
    doc_box_color_text = []

    for idx in range(np.random.randint(5, max_phrase)):
        item = texts[np.random.randint(len(texts))]
        item_str = list(item)
        # print('item str :', item_str)
        _text = item_str.copy()
        _box_color_text= item_str.copy()
        
        # for idx in range(np.random.randint(1,3)):
        #  随机插入latex到文本中
        pos = np.random.randint(1,len(_text))
        latex ='${}$'.format(latexs[np.random.randint(len(latexs))])
        box_color_latex = '\colorbox{%s}{%s}' % (latex_box_color,latex)
        latex = '\colorbox{white}{%s}' % (latex)
        _text.insert(pos, latex)
        _box_color_text.insert(pos, box_color_latex)

        # 随机在文本中插入图片
        if np.random.randint(3) == 0:
            pos = np.random.randint(1,len(_text))
            insert_pic_file = pic_files[np.random.randint(len(pic_files))]
            picture = '\\fcolorbox{white}{white}{\includegraphics[scale=0.2]{%s}}' % (insert_pic_file)
            color_picture = '\\fcolorbox{blue}{blue}{\includegraphics[scale=0.2]{%s}}' % (insert_pic_file)
            _text.insert(pos, picture)
            _box_color_text.insert(pos, color_picture)


        doc_text.append(''.join(_text))
        doc_box_color_text.append(''.join(_box_color_text))


    # 随机在文本段中插入图片
    # if np.random.randint(2) == 0:
    insert_pic_file = pic_files[np.random.randint(len(pic_files))]
    pos = np.random.randint(1, len(doc_text))
    picture = '\\begin{figure}[ht] \centering \\fcolorbox{white}{white}{\includegraphics[scale=0.5]{%s}} \end{figure}' % insert_pic_file
    color_picture = '\\begin{figure}[ht] \centering \\fcolorbox{blue}{blue}{\includegraphics[scale=0.5]{%s}} \end{figure}' % insert_pic_file

    # print(color_picture)    

    doc_text.insert(pos, picture)
    doc_box_color_text.insert(pos, color_picture)

    # print(doc_box_color_text)

    gen_pdf(data_root, file_name, doc_text)
    gen_pdf(data_root, f'{file_name}_color', doc_box_color_text)        

def gen_pdf(data_root, file_name, texts):
    file_path = os.path.sep.join([data_root,'pdf', file_name])
    if os.path.exists(file_path):
        os.remove('{}.pdf'.format(file_path))

    doc = Document()
    doc.packages.add(Package('ctex'))
    doc.packages.add(Package('color'))
    doc.packages.add(Package('xcolor'))  
    doc.packages.add(Package('graphicx'))  
    doc.packages.add(Package('geometry'))  
    doc.append(NoEscape(r'\newgeometry{left=3cm,bottom=1cm}'))

    with doc.create(Section('训练数据')):
        for t in texts:
            doc.append(NoEscape(r'%s' % (t)))
            doc.append(NoEscape(r'\linebreak'))

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
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    formula_pos = []

    # tmpImg = image.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(tmpImg, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # formula_pos append (x1,y1,x2,y2, math label)
        formula_pos.append([x,y,x+w,y+h,0])

    # print(tmpImg.shape)
    # print(np.array(formula_pos))

    # plt.imshow(tmpImg)
    # plt.show()

    np.savetxt(os.path.sep.join([anno_dir, f'{file_name}.pmath']),np.array(formula_pos),'%.3f', ',', )


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


    np.savetxt(os.path.sep.join([anno_dir, f'{file_name}.ppic']),np.array(pic_pos),'%.3f', ',', )


    # PDF生成图片
    # with open(f'{pdf_file_path}_color.pdf','rb') as f:
    #     data = f.read()
    # image = pdf2image(data, imgH=imgH)
    # image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
    # img_dir = os.path.sep.join([data_root,'data', image_dir, sub_dir])
    # if not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    # cv2.imwrite(os.path.sep.join([img_dir, f'{file_name}_color.png']), image)

    with open(f'{pdf_file_path}.pdf','rb') as f:
        data = f.read()
    image = pdf2image(data, imgH=imgH)
    image = cv2.imdecode(np.frombuffer(image, np.uint8),cv2.IMREAD_COLOR)
    img_dir = os.path.sep.join([data_root,'data', image_dir, sub_dir])
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    cv2.imwrite(os.path.sep.join([img_dir, f'{file_name}.png']), image)


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






