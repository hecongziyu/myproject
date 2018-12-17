# -*- coding:utf-8 -*-
# 生成测试Image
import lib.data.char as char
from PIL import Image, ImageDraw, ImageFont
import random

def genImage(label, 
            fontsize, 
            color=(0, 0, 0),
            fontName=None,
            data_path=None):
    font_path = '{}{}.ttf'.format(data_path,fontName)
    data_img_path = '{}dataline/'.format(data_path)
    img = Image.new("RGB", ((int)(fontsize * 1.2 * len(label)), (int)(fontsize * 2)), (255, 255, 255))
    font = ImageFont.truetype(font_path, fontsize)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), label, fill=color, font=font)
    with open('{}{}-{}-{}.txt'.format(data_img_path,fontName, str(fontsize),label), "w", encoding='utf-8') as f:
        f.write(label)
    img.save('{}{}-{}-{}.jpg'.format(data_img_path , fontName, str(fontsize),label))


def create_data(number, font_name, data_path):
    alphabet = char.alphabet
    charact = alphabet[:]
    textLen = len(charact) - 11
    for i in range(number):
        ss = random.randint(0, textLen)
        genImage(alphabet[ss:ss + 10], 20, fontName=font_name, data_path=data_path)
        genImage(alphabet[ss:ss + 10], 15, fontName=font_name, data_path=data_path)


if __name__ == '__main__':
    path = 'D:/PROJECT_TW/git/data/ocr/'
    font_name = 'stsong'
    create_data(10,font_name=font_name, data_path=path)






    
