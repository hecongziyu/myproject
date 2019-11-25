# encoding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
def detect_char_area(image_gray_data, min_area = 80, min_y_diff=10):
    img = image_gray_data.copy()
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    img = cv2.erode(img, element,iterations=1)
    img = cv2.dilate(img, element,iterations=3)
    # plt.imshow(img,'gray')
    # plt.show()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:    
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            cnts.append([x,y,x+w,y+h, cv2.contourArea(cnt)])
    areas = np.array(cnts,dtype=np.uint8)
    areas_max = np.argmax(areas[:,4], axis=0)
    x1,y1,x2,y2,_ = areas[areas_max]
    cent_y = y1 + int((y2-y1)/2)
    areas_cents = [int((x[3] - x[1])/2 + x[1])  for x in areas]
    areas_filter_idx = np.where(abs(areas_cents - cent_y) < min_y_diff)
    x1,y1,x2,y2 = np.min(areas[areas_filter_idx,0]),np.min(areas[areas_filter_idx,1]),np.max(areas[areas_filter_idx,2]),np.max(areas[areas_filter_idx,3])
    x1 = max(x1-1,0)
    y1 = max(y1-1,0)
    x2 = min(x2+1,img.shape[1])
    y2 = min(y2+1,img.shape[0])
    return x1,y1,x2,y2