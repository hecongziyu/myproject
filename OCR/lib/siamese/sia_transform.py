import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class RandomExpand(object):
    def __init__(self):
        pass
    def __call__(self,image):
        pass

class RandomSize(object):
    def __init__(self, min_radio=0.5, max_radio=1.2):
        self.min_radio = min_radio
        self.max_radio = max_radio

    def __call__(self, image):
        height, width, _ = image.shape
        random_sel = np.random.randint(3)
        if random_sel == 0:
            # 等比例缩放
            min_radio = self.min_radio
            max_radio = self.max_radio
            if height < 100 or width < 100:
                min_radio = 1.0
                max_radio = 1.4
            elif height > 800 or height > 800:
                min_radio = 0.3
            radio = random.uniform(min_radio, max_radio)
            image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_AREA)
        elif random_sel == 1:
            # 只缩放宽度
            radio = random.uniform(0.8, self.max_radio)
            image = cv2.resize(image.copy(), (int(width*radio),int(height)), interpolation=cv2.INTER_AREA)
        else:
            # 只缩放高度
            radio = random.uniform(0.8, self.max_radio)
            image = cv2.resize(image.copy(), (int(width),int(height * radio)), interpolation=cv2.INTER_AREA)
        return image

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=0.7):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if np.random.randint(4)==0:
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image

class StructResize(object):
    # 调整到训练用的图片， 宽高均为size, 用于CNN训练, 注意这里不再调整boxes的大小
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image.copy(), (self.size,self.size), interpolation=cv2.INTER_AREA)
        return image

class ConvertFromInts(object):
    def __call__(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # return image.astype(np.float32)
        return image

class LmdbTransform(object):
    def __init__(self, size=400):
        self.size = size
        self.augment = Compose([
            # RandomExpand(),
            RandomSize(min_radio=0.5, max_radio=1.2),
            ConvertFromInts(),
            # RandomContrast(),
            StructResize(size=self.size)
        ])

    def __call__(self, img):
        return self.augment(img)    
