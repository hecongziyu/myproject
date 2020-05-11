# -*- coding: UTF-8 -*-
'''
https://pymupdf.readthedocs.io/en/latest/
'''
import fitz
import numpy as np
import random
import cv2
import lib.im2latex.gen_latex_img as lxu
import os

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


