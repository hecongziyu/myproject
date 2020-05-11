# -*- coding: UTF-8 -*-
from pytexit import py2tex
from random import randint
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt 
import io
import cv2

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
    plt.savefig(buf,format='png',transparent=False,pad_inches=0,dpi=600)
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





