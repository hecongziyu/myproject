# -*- coding:utf-8 -*-
# 生成测试Image
import lib.data.char as char
from PIL import Image, ImageDraw, ImageFont
import random
data_path = '/home/hecong/temp/data/ocr/dataline/'
def genImage(label, fontsize, color=(0, 0, 0),fontName="/home/hecong/temp/data/ocr/华文细黑.ttf"):
    img = Image.new("RGB", ((int)(fontsize * 1.2 * len(label)), (int)(fontsize * 2)), (255, 255, 255))
    font = ImageFont.truetype(fontName, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), label, fill=color, font=font)
    with open(data_path + label + "-" + str(fontsize) + ".txt", "w", encoding='utf-8') as f:
        f.write(label)
    img.save(data_path + label + "-" + str(fontsize) + ".jpg")


def create_data(number):
    alphabet = char.alphabet
    charact = alphabet[:]
    textLen = len(charact) - 11
    for i in range(number):
        ss = random.randint(0, textLen)
        genImage(alphabet[ss:ss + 10], 20)
        genImage(alphabet[ss:ss + 10], 15)
    
