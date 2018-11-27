# -*- coding:utf-8 -*-
# 生成测试Image
from PIL import Image, ImageDraw, ImageFont
import random

def genImage(label, fontsize, color=(0, 0, 0),fontName=u"simsunb.ttf"):

    print('label {} fontsize {}'.format(label, fontsize))
    
    img = Image.new("RGB", ((int)(fontsize/1.65 * len(label)), (int)(fontsize * 2)), (255, 255, 255))
    print(((int)(fontsize * len(label)), (int)(fontsize * 2)))
    font = ImageFont.truetype(fontName, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text((5, 10), label, fill=color, font=font)
    with open("./data/dataline/" + label + "-" + str(fontsize) + ".txt", "w", encoding='utf-8') as f:
        f.write(label)
    img.save("./data/dataline/" + label + "-" + str(fontsize) + ".jpg")
