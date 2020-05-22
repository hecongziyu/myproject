import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import os

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

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        # print('penc image shape:', image.shape)
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels    

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        # if random.randint(2):
        #     return image, boxes, labels
        # print('input boxes:', boxes)

        height, width, depth = image.shape
        ratio = random.uniform(1, 3)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)
        expand_image = np.ones((int(height*ratio), int(width*ratio), depth),dtype=image.dtype) * 255
        # expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        # print('out put boxes:', boxes)
        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        # print('resize img shape:', image.shape)
        image = cv2.resize(image, (self.size,self.size), interpolation=cv2.INTER_AREA)
        return image, boxes, labels

'''替换前景景图片'''
class BackGround(object):
    def __init__(self, data_root, back_data_dir='bg'):
        self.data_root = data_root
        self.back_data_dir = back_data_dir

    def __call__(self, image,boxes=None,labels=None):
        if random.randint(3) == 0:
            return image,boxes,labels    
        height,width, _ = image.shape
        bg_files = os.listdir(os.path.sep.join([self.data_root,self.back_data_dir]))
        # print('bg files:',bg_files)
        bg_img_file = os.path.sep.join([self.data_root, self.back_data_dir, bg_files[random.randint(0, len(bg_files))]])
        # print('bg img file:', bg_img_file)
        bg_img = cv2.imread(bg_img_file,cv2.IMREAD_UNCHANGED)
        bg_img = cv2.resize(bg_img, (width,height), interpolation=cv2.INTER_AREA)
        bg_img[np.where(image<=128)] = 0
        return bg_img,boxes,labels


class GTDBTransform(object):
    def __init__(self, data_root, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.data_root = data_root
        self.augment = Compose([
            BackGround(data_root=self.data_root),
            ConvertFromInts(),
            # ToAbsoluteCoords(),  # ToAbsoluteCoords 转成绝对坐标，生成的box进行了缩放
            # ToPercentCoords(),   # 与ToAbsoluteCoords对应，将target 恢复成x1/width, x2/width, y1/height, y2/height
            Resize(self.size)   # 将变换后图片转成 size * size
            # SubtractMeans(self.mean) 
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
