import os
import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import random
import json
import numpy as np

def make_image(test_str, fonttype, fontsize,target_width ,target_height, 
               back_ground_img_list, blurs=((3,3),(4,4),(5,5)), 
               keep_redio=True,
               need_include=False):
    font_type = ImageFont.truetype(fonttype, fontsize)
    text_width, text_height = font_type.getsize(test_str)
    pd = 20
    bg_img = cv2.imread(back_ground_img_list[np.random.randint(len(back_ground_img_list))],cv2.IMREAD_COLOR)
    bg_img = cv2.resize(bg_img, (text_width+pd, text_height+2), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB)) 

    draw = ImageDraw.Draw(image)
    draw_str = list(test_str)
    draw_str = ''.join(draw_str)
    


    draw.text((pd/2, 0), draw_str, (0,0,0), font=font_type)
        
    #绘制噪点
    if np.random.randint(10)<7:
        for i in range(0, 20):
            xy = (random.randrange(0, text_width+pd), random.randrange(0, text_height+2))
            if i%2 == 0:
                fill_color = (255, 255, 255)
            else:
                fill_color = (0, 0, 0)
            draw.point(xy, fill=fill_color)
        
        
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
#     image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)     
    #  图像模糊
    if np.random.randint(10)<7:
#         image = cv2.GaussianBlur(image,(0,0),round(np.random.uniform(0.90,1),2))    
        image = cv2.blur(image,(np.random.randint(2,5), np.random.randint(2,5)))

    open_cv_image = image.copy()
#     if keep_redio:
#         percent = float(32) / open_cv_image.shape[0]
#         open_cv_image = cv2.resize(open_cv_image,(0,0), fx=percent, fy=percent, interpolation = cv2.INTER_AREA)

    return open_cv_image


def gen_words(file_name, alphas, max_number=6, total=1000, noise_alpha=None,need_include=True):
    with open(file_name, 'w') as f:
        for _ in range(total):
            line = []
#             for _ in range(np.random.randint(max_number)+1):
            for _ in range(max_number):
                line.append(alphas[np.random.randint(len(alphas))])
                if noise_alpha and np.random.randint(3)==1:
                    line.append(noise_alpha[np.random.randint(len(noise_alpha))])
            line.append('\n')
            line = ''.join(line)
            f.write(line)