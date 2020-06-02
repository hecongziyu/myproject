# encoding: utf-8
import cv2
import numpy as np
# import matplotlib.pyplot as plt

def detect_lines(image_gray_data, min_line_length=None, max_line_gap=10):
    ret, img = cv2.threshold(image_gray_data, 140, 255, cv2.THRESH_BINARY_INV)
    cols = img.shape[1]

    horizontal_size = int(cols / 20)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    img = cv2.erode(img, horizontalStructure)
    img = cv2.dilate(img, horizontalStructure)
    if min_line_length:
        minLineLength = min_line_length
    else:
        minLineLength = int(img.shape[1] * 0.1)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, minLineLength=minLineLength, maxLineGap=max_line_gap)
    image_lines = []

    if lines is not None:
        lines = np.array(lines)
        lines = lines.reshape(-1, 4)
        for line in lines:
            x1, y1, x2, y2 = line
            image_lines.append([x1,y1,x2,y2])
    else:
        image_lines.append([0,0,0,0])
    image_lines = np.array(image_lines)
    # 根据Y轴进行排重
    va, inx = np.unique(image_lines[:, 1] + image_lines[:, 0], return_index=True)
    image_lines = image_lines[inx]
    idex = np.lexsort([image_lines[:, 0], image_lines[:, 1]])
    image_lines = image_lines[idex, :]
    return image_lines.tolist()

'''
检测图片文字区域，返回文字最大一行的位置信息
image_gray_data 灰度图片
min_area_radio  文字区域最小区域
'''
def detect_char_area(image_gray_data, min_area = 80,min_y_diff=5, cent_y=None):
    img = image_gray_data.copy()
    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=3)    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:    
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            if (y+h)/2 > img.shape[0]*0.1:
                cnts.append([x,y,x+w,y+h, cv2.contourArea(cnt)])
    areas = np.array(cnts,dtype=np.uint8)
    # if areas is None or len(areas) == 0:
    #     return 0,0,0,0
    # areas_max = np.argmax(areas[:,4], axis=0)
    # x1,y1,x2,y2,_ = areas[areas_max]

    # print('len areas:', len(areas))
    # print('areas:', areas)
    # print('cent_y:', cent_y)

    # if cent_y is None:
    #     cent_y = y1 + int((y2-y1)/2)
    
    # areas_cents = [int((x[3] - x[1])/2 + x[1])  for x in areas]
    # areas_filter_idx = np.where(abs(areas_cents - cent_y) < min_y_diff)
    x1,y1,x2,y2 = np.min(areas[:,0]),np.min(areas[:,1]),np.max(areas[:,2]),np.max(areas[:,3])
    return x1,y1,x2,y2


def detect_char_pos(image_gray_data, min_area = 80,min_y_diff=5):
    img = image_gray_data.copy()
    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=3)    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:    
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            if (y+h)/2 > img.shape[0]*0.1:
                cnts.append([x,y,x+w,y+h, cv2.contourArea(cnt)])
    areas = np.array(cnts,dtype=np.uint8)
    if areas is None or len(areas) == 0:
        return 0,0,0,0
    areas_max = np.argmax(areas[:,4], axis=0)
    x1,y1,x2,y2,_ = areas[areas_max]
    
    # cent_y = y1 + int((y2-y1)/2)
    # areas_cents = [int((x[3] - x[1])/2 + x[1])  for x in areas]
    # areas_filter_idx = np.where(abs(areas_cents - cent_y) < min_y_diff)
    # x1,y1,x2,y2 = np.min(areas[areas_filter_idx,0]),np.min(areas[areas_filter_idx,1]),
    #                     np.max(areas[areas_filter_idx,2]),np.max(areas[areas_filter_idx,3])

    # y_bias = int(img.shape[0]*0.1)
    # x1 = min(x1,1)
    # y1 = max(y1-y_bias,0)
    # x2 = max(x2,img.shape[1] - 1)
    # y2 = min(y2+y_bias,img.shape[0])
    return x1,y1,x2,y2


'''
    将原图转成二值化, 黑底白字
    need_dilate 
'''
def convert_img_bin(image_gray_data, thread_pre=15, thread_post=10, need_dilate=False):
    gray = image_gray_data.copy()
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    img  = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,thread_pre,thread_post)  
    if need_dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        img = cv2.dilate(img, kernel, iterations=2)
    return img 

