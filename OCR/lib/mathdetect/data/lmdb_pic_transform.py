import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 5)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)
    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, element1, iterations = 1)
    # 6. 再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, element2, iterations = 1)
    return dilation

def findTextRegion(img):
    region = []
    height,width  = img.shape
 
    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    box_lists = []
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 

        # 面积小的都筛选掉
        if(area < 30):
            continue
 
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
 
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        x0 = np.min(box[:,0])
        x1 = np.max(box[:,0])
        y0 = np.min(box[:,1])
        y1 = np.max(box[:,1])

        box_lists.append([x0,y0,x1, y1])
        # 计算高和宽
        # height = abs(box[0][1] - box[2][1])
        # width = abs(box[0][0] - box[2][0])
 
        # # 筛选那些太细的矩形，留下扁的
        # if(height > width * 1.2):
        #     continue
        # region.append(box)
    box_lists = np.array(box_lists)
    # print('boxes lists :', box_lists)
    x0 = max(np.min(box_lists[:,0]) - 5,0)
    y0 = max(np.min(box_lists[:,1]) - 5,0)
    x1 = min(np.max(box_lists[:,2]) + 5, width)
    y1 = min(np.max(box_lists[:,3]) + 5, height)

    box = (x0,y0,x1,y1)

    return box




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

# 去文字白边
class RemoveWhiteBoard(object):
    def __call__(self, image, boxes, labels):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = preprocess(gray)

 
        # 3. 查找和筛选文字区域
        text_boxes = findTextRegion(dilation)
        if text_boxes is not None:
            x0,y0,x1, y1 = text_boxes
            rimage = image[y0:y1, x0:x1]
            boxes[:,(0,2)] -= x0
            boxes[:,(1,3)] -= y0
            return rimage, boxes, labels
        else:
            return image, boxes, labels




class RandomSize(object):
    def __init__(self, min_radio=0.6, max_radio=1):
        self.min_radio = min_radio
        self.max_radio = max_radio

    def __call__(self, image,boxes=None, labels=None):
        # if random.randint(2):

        radio = random.uniform(self.min_radio, self.max_radio)
        height, width, _ = image.shape
        if width > 200 and height > 80:
            image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_AREA)
            boxes = boxes * radio
        return image,boxes, labels

class AdjustSize:
    '''
    调整图片大小，图片宽度不能超过设定最大宽度
    '''
    def __init__(self, max_width=300):
        self.max_width = max_width

    def __call__(self, image, boxes=None, labels=None):
        # print('resize img shape:', image.shape)
        if image.shape[1] > self.max_width or image.shape[0] > self.max_width:
            image = image.astype(np.uint8)
            radio = min(self.max_width / image.shape[1], self.max_width / image.shape[0])
            image = cv2.resize(image.copy(), (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
            boxes = boxes * radio
        return image, boxes, labels


class Mask2Windows:
    # 将图片贴在设定的窗口里面
    def __init__(self, window):
        self.window = window

    def __call__(self, image, boxes=None, labels=None):
        win_img = np.full((self.window, self.window, image.shape[2]), 255)
        # if random.randint(3) == 0:
        win_img[0:image.shape[0],0:image.shape[1],:] = image.copy()
        # else:
        #     # 原图在window上面随机偏移位置
        #     xl = random.randint(self.window - image.shape[1])
        #     yl = random.randint(0, min(20,self.window - image.shape[0]))
        #     win_img[yl:yl+image.shape[0], xl:xl+image.shape[1],:] = image.copy()
        #     boxes = boxes.copy()
        #     boxes[:,(0,2)] += xl
        #     boxes[:,(1,3)] += yl
        return win_img, boxes, labels


class StructResize(object):
    # 调整到训练用的图片， 宽高均为size, 用于CNN训练, 注意这里不再调整boxes的大小
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image.copy(), (self.size,self.size), interpolation=cv2.INTER_AREA)
        return image, boxes, labels    

class FormulaTransform(object):
    def __init__(self, window=1200, max_width=1200, size=300):
        self.window = window
        self.max_width = max_width
        self.size = size
        self.augment = Compose([
            # RemoveWhiteBoard(),
            # RandomSize(min_radio=0.9, max_radio=1),
            AdjustSize(max_width=self.max_width),
            Mask2Windows(window=self.window),
            ConvertFromInts(),
            StructResize(size=self.size)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)    

class PicTransform(object):
    def __init__(self, window=1200, max_width=1200, size=300):
        self.window = window
        self.max_width = max_width
        self.size = size
        self.augment = Compose([
            # RemoveWhiteBoard(),
            # RandomSize(min_radio=0.9, max_radio=1),
            AdjustSize(max_width=self.max_width),
            Mask2Windows(window=self.window),
            ConvertFromInts(),
            StructResize(size=self.size)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)            


