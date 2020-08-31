import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, qr_img, boxes=None):
        for t in self.transforms:
            img, qr_img, boxes = t(img, qr_img, boxes)
        return img, qr_img, boxes



# 随机粘贴二维码图片到底片上面
class MaskQRImage(object):
    def __call__(self, img, qr_img, boxes):
        height, width, _ = img.shape
        q_height, q_width, _ = qr_img.shape
        for idx in range(np.random.randint(0,5)):
            x_pos = np.random.randint(5, width-q_width-5)
            y_pos = np.random.randint(5, height-q_height-5)
            img[y_pos:y_pos+q_height, x_pos:x_pos+q_width, :] = qr_img
            
            boxes = np.r_[boxes,np.array([[x_pos,y_pos,x_pos+q_width,y_pos+q_height, 0]])]

        # if idx > 1:
        if boxes.shape[0] > 1:
            boxes = boxes[np.where(boxes[:,4] != -1)]
        return img,qr_img, boxes

class Mask2Windows:
    # 将图片贴在设定的窗口里面
    def __init__(self, window):
        self.window = window

    def __call__(self, img, qr_img, boxes):
        win_img = np.full((self.window, self.window, img.shape[2]), 255)
        # if random.randint(3) == 0:
        win_img[0:img.shape[0],0:img.shape[1],:] = img.copy()
        # else:
        #     # 原图在window上面随机偏移位置
        #     xl = random.randint(self.window - img.shape[1])
        #     yl = random.randint(0, min(20,self.window - img.shape[0]))
        #     win_img[yl:yl+img.shape[0], xl:xl+img.shape[1],:] = img.copy()
        #     boxes = boxes.copy()
        #     boxes[:,(0,2)] += xl
        #     boxes[:,(1,3)] += yl
        # return win_img, qr_img, boxes
        return win_img, qr_img, boxes

class AdjustSize:
    '''
    调整图片大小，图片宽度不能超过设定最大宽度
    '''
    def __init__(self, max_width=1200):
        self.max_width = max_width

    def __call__(self, img, qr_img, boxes):
        # print('resize img shape:', image.shape)
        if img.shape[1] > self.max_width or img.shape[0] > self.max_width:
            img = img.astype(np.uint8)
            radio = min(self.max_width / img.shape[1], self.max_width / img.shape[0])
            img = cv2.resize(img.copy(), (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
            boxes[:,0:4] = boxes[:,0:4] * radio
        return img, qr_img, boxes      


class StructResize(object):
    # 调整到训练用的图片， 宽高均为size, 用于CNN训练, 注意这里不再调整boxes的大小
    def __init__(self, size):
        self.size = size

    def __call__(self, img, qr_img, boxes):
        img = img.astype(np.uint8)
        img = cv2.resize(img.copy(), (self.size,self.size), interpolation=cv2.INTER_AREA)
        return img, qr_img, boxes

class ConvertFromInts(object):
    def __call__(self, img, qr_img, boxes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.astype(np.float32), qr_img, boxes


class QRTransform(object):
    def __init__(self, window=1200, max_width=1200, size=300):
        self.window = window
        self.max_width = max_width
        self.size = size
        self.augment = Compose([
            MaskQRImage(),
            AdjustSize(),
            Mask2Windows(window=self.window),
            StructResize(size=self.size),
            ConvertFromInts()

        ])

    def __call__(self, img, qr_img, boxes):
        return self.augment(img, qr_img, boxes)    



class QRTestTransform(object):
    def __init__(self, window=1200, max_width=1200, size=300):
        self.window = window
        self.max_width = max_width
        self.size = size
        self.augment = Compose([
            MaskQRImage(),
            # AdjustSize(),
            # Mask2Windows(window=self.window),
            # StructResize(size=self.size),
            # ConvertFromInts()
        ])

    def __call__(self, img, qr_img, boxes):
        return self.augment(img, qr_img, boxes)    
