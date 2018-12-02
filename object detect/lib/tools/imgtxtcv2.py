# -*- coding:utf-8 -*-
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

# https://blog.csdn.net/it2153534/article/details/79185397

def get_img_text_box(img_name):
    img_boxes = []

    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 此步骤形态学变换的预处理，得到可以查找矩形的图片
    # 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 1)

    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 设置膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 3))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
    # 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用

    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)

    # cv2.imshow('dilation',dilation)
    # cv2.waitKey(0)
    # plt.imshow(erosion,'gray')
    # plt.show()

    dilation2 = dilation
    #  查找和筛选文字区域
    region = []
    #  查找轮廓
    img2, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #  查找和筛选文字区域
    region = []
    #  查找轮廓
    img2, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
    #  筛选那些面积小的
    for i in range(len(contours)):
        # 遍历所有轮廓
        # cnt是一个点集
        cnt = contours[i]

        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉、这个1000可以按照效果自行设置
        if(area < 400):
            continue

        #     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
        #     # 轮廓近似，作用很小
        #     # 计算轮廓长度
        #     epsilon = 0.001 * cv2.arcLength(cnt, True)

        #     #
        # #     approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # 打印出各个矩形四个点的位置
        # print ("rect is: ")
        # print (rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)



        # # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        #
        # # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.3):
            continue
        box = box.reshape(-1)
        img_boxes.append(box.tolist())
        #
        # region.append(box)

    # 用绿线画出这些找到的轮廓
    # for box in region:
    #     cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # ",".join(map(str,aa))
    # plt.imshow(img, 'brg')
    # plt.show()
    return img_boxes

def show_img_box(img_name, in_boxes):
    img = cv2.imread(img_name)
    region = []
    for item in in_boxes:
        box = np.array(item)
        box = box.reshape(4,-1)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        if (height > width * 1.3):
            continue
        region.append(box)
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 1)

    cv2.imshow("img",img)
    cv2.waitKey(0)

def split_box_size(in_boxes, width=16):
    out_boxes = []
    print(in_boxes)
    for box in in_boxes:
        box = np.array(box)
        box = box.reshape(-1,2)

        box = box[box[:,0].argsort()]
        start_box  = box[:2]
        end_box = box[2:]
        number = int((end_box[0][0] - start_box[0][0])/width)

        # import ipdb
        # ipdb.set_trace()
        # 按最后一列降序
        start_box=start_box[np.lexsort(-start_box.T)]
        old_box = start_box
        for i in np.arange(1,(1+number),1):
            tmp_box = start_box + ([width*i,0])
            tmp_box = tmp_box[tmp_box[:,1].argsort()]
            new_box = np.concatenate((old_box,tmp_box))
            old_box = tmp_box[np.lexsort(-tmp_box.T)]
            out_boxes.append(new_box.reshape(-1).tolist())

    print(out_boxes)
    return out_boxes

def save_box(file, box):
    with open(file, 'w+') as fw:
        for item in box:
            fw.write(",".join(map(str,item))+"\n")






if __name__ == '__main__':
    img_box = get_img_text_box("d:\\cc_3.jpg")
    out_box = split_box_size(img_box)
    save_box("d:\\1.txt",out_box)
    # mb = [16,0]
    # tmp_box = np.array(img_box[0])
    # tmp_box = tmp_box.reshape(-1,2)
    # start_box = tmp_box[:2]
    # end_box = tmp_box[2:]
    # number = int(end_box[0][0] / start_box[0][0])
    show_img_box("d:\\cc_3.jpg",img_box)
