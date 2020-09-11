import torch
from torchvision import transforms
import cv2
import numpy as np
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

# 去文字白边
class RemoveWhiteBoard(object):
    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = preprocess(gray)

        # 3. 查找和筛选文字区域
        text_boxes = findTextRegion(dilation)
        if text_boxes is not None:
            x0,y0,x1, y1 = text_boxes
            rimage = image[y0:y1, x0:x1]
            return rimage
        else:
            return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)

# 随机改变大小
class RandomSize(object):
    def __init__(self, min_radio=0.35, max_radio=0.5, min_size=900, min_height=24):
        self.min_radio = min_radio
        self.max_radio = max_radio
        self.min_height = min_height
        self.min_size = 900
    def __call__(self, image):
        height, width, _ = image.shape
        min_radio = self.min_radio
        max_radio = self.max_radio
        # if width > 900 or height > 900:
        #     min_radio = 0.05
        #     max_radio = 0.15
        # elif width > 700 or height > 700:
        #     min_radio = 0.1
        #     max_radio = 0.2
        # elif width > 400 or height > 400:
        #     min_radio = 0.15
        #     max_radio = 0.25
        # elif width > 250  or height > 250:
        #     min_radio = 0.2
        #     max_radio = 0.3


        radio = random.uniform(min_radio, max_radio)
        radio = max(radio, self.min_height/height)
        print('befor image shape :', image.shape, ' radio :', radio)
        image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_AREA)
        print('after image shape ', image.shape)
        return image

class AdjustSize(object):
    '''
    调整图片大小，图片宽度不能超过设定最大宽度
    '''
    def __init__(self, max_height=400, max_width=600):
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, image):
        # print('resize img shape:', image.shape)
        height, width, _ = image.shape
        if height > self.max_height or width > self.max_width:
            radio = min(self.max_width / width, self.max_height / height)
            image = cv2.resize(image.copy(), (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
        return image



class Mask2Windows:
    # 将图片贴在设定的窗口
    def __init__(self, window):
        self.window = window

    def __call__(self, image):
        height, width = self.window

        win_img = np.full((height, width, image.shape[2]), 255)
        if random.randint(2) == 0:
            win_img[0:image.shape[0],0:image.shape[1],:] = image.copy()
        else:
        #     # print('width :', width,  ' image shape :', image.shape[1])
        #     # print('height :', height,  ' image shape :', image.shape[0])
            x1 = 0
            y1 = 0
            if width > image.shape[1]:
                x1 = random.randint(0, min(20,width - image.shape[1]))
            if height > image.shape[0]:
                y1 = random.randint(0, min(20,height - image.shape[0]))

            win_img[y1:y1+image.shape[0], x1:x1+image.shape[1],:] = image.copy()

        return win_img






class ImgTransform(object):
    def __init__(self, window=(300,500), max_height=300, max_width=500, min_size=900):
        self.max_width = max_width
        self.max_height = max_height
        self.min_size = min_size
        self.window = window

        self.augment = Compose([
                RemoveWhiteBoard(),
                # 注意因为生成训练图片时，图片是按8096/5左右大小生成的，所以图片需要放小
                RandomSize(min_size=self.min_size, min_radio=0.9, max_radio=1),
                # AdjustSize(max_height=self.max_height, max_width=self.max_width),
                # Mask2Windows(window=self.window),
                ConvertFromInts()
            ])


    def __call__(self, image):
        return self.augment(image)



