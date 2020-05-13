# -*- coding: UTF-8 -*-
from pytexit import py2tex
from random import randint
from PIL import Image
from build_vocab import build_vocab
import numpy as np 
from matplotlib import pyplot as plt 
import io
import cv2
import os
import random

'''
生成latex 图像

显示 matplotlib 的字体
import matplotlib
a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
print(a)

https://matplotlib.org/tutorials/text/mathtext.html

'''

fms = ['\\frac {{ {0} }} {{ {1} }} + {{ {2} }}', 
        '\\sqrt {{ {0} {1} }}',
        '{0} ^ 2',
        '{0} ^ 3',
        '{{ {0} }} \\times {{ {1} }} - {2} {3}', 
       '{0} - \\frac {{ {1} ^ {2} }} {{ {3} }} = {4}',
       '\\sqrt {{ {0} {1} }} + \\frac {{ {3} }} {{ {4} }}',
       '{0} \\div {1} \\times \sqrt {{ {2} }} = {3}']

# size 生成数量
def random_latex(size=1):
	formul_lists = []
	for idx in range(size):
		fm = fms[randint(0,len(fms)-1)]
		fm = fm.format(randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9))
		formul_lists.append(fm)
	return formul_lists


# latex font 字体设置 
# https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
# https://stackoverflow.com/questions/17958485/matplotlib-not-using-latex-font-while-text-usetex-true
def latex_to_img(latex_text,family='Courier New', fontsize=40):
	# 需设置figsize，否则latex可能会超过图片大小
    fig = plt.figure(figsize=(8,4))
    buf = io.BytesIO()
    plt.axis('off')
    # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
    plt.rc('mathtext',fontset='cm')    
    plt.rc('text', usetex=False)
    plt.rc('font', family=family)
    plt.text(0.5, 0.5, r"$%s$" % latex_text,fontsize = fontsize, ha='center', va='center')
    # plt.savefig(buf,format='png',transparent=False,pad_inches=0,dpi=300)
    plt.savefig(buf,format='png',transparent=False,pad_inches=0)
    plt.close()
    image = Image.open(buf)
    return image


def image_ract(image_array):
    x_array, y_array = np.where(image_array==1)
    return x_array[np.argmin(x_array)] - 40 , y_array[np.argmin(y_array)] - 40 , x_array[np.argmax(x_array)] + 40, y_array[np.argmax(y_array)] + 40


def get_latex_image(latex_text):
    image = latex_to_img(latex_text)
    image_array = np.array(image.copy().convert("L"))
    image_array_new = 1 - image_array/255 
    x1,y1,x2,y2 = image_ract(image_array_new)
    image = image.crop((y1,x1,y2,x2))
    image = image.convert('RGB')
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    return image


def gen_latex_formula_batch(data_root,size=1000,file_name='latex_formul_normal.txt', img_dir='gen_images'):
    formul_lists = []
    

    # 生成公式文本
    formul_files = os.path.sep.join([data_root, file_name])
    for idx in range(len(fms),len(fms)+size):
        fm = fms[idx % len(fms)]
        fm = fm.format(randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9),randint(0,9))
        fm = fm + '\n'
        formul_lists.append(fm)
    with open(formul_files, 'w', encoding='utf8') as f:
        f.writelines(formul_lists)


    # 生成公式图片
    for idx in range(len(formul_lists)):
        tex = formul_lists[idx]
        tex = tex.replace('\n','')
        if idx % 1000 == 0:
            print('gen image :', idx)
        image = get_latex_image(tex)
        cv2.imwrite(os.path.sep.join([data_root,img_dir,f'{idx}.png']),image)

    # 数据拆分成训练、验证、测试三类数据
    train_split_idx = random.sample(range(len(formul_lists)), int(len(formul_lists)*0.8))
    valid_test_data = [x for x in range(len(formul_lists)) if x not in train_split_idx]
    valid_split = random.sample(valid_test_data, int(len(valid_test_data)*0.5))
    test_split = [x for x in valid_test_data if x not in valid_split]

    with open(os.path.sep.join([data_root, 'latex_{}_filter.txt'.format('train')]), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(['{}.png {}'.format(x,x) for x in train_split_idx]))

    with open(os.path.sep.join([data_root, 'latex_{}_filter.txt'.format('test')]), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(['{}.png {}'.format(x,x) for x in test_split]))

    with open(os.path.sep.join([data_root, 'latex_{}_filter.txt'.format('valid')]), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(['{}.png {}'.format(x,x) for x in valid_split]))


    build_vocab(data_root, min_count=5)

    # 生成vocab字典

    print('生成训练数据完成')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex train data')
    parser.add_argument('--dataset_root', default='D:\\PROJECT_TW\\git\\data\\im2latex',
                        help='data set root')
    parser.add_argument('--gen_size', default='100',help='gen size')    
    args = parser.parse_args()

    gen_latex_formula_batch(data_root=args.dataset_root, size=int(args.gen_size))




