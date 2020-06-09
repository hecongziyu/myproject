# encoding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from lmdb_dataset import BLANK_FLAG
import random
import os

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, char_img, bg_image=None, label=None):
        for t in self.transforms:
            char_img, bg_image, label = t(char_img, bg_image, label)
        return char_img, bg_image, label

class ConvertFromInts(object):
    def __call__(self, image, bg_image=None, label=None):
        if len(image.shape) == 3:
            # 转换成灰度图
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # print('image shape:', image.shape)
        return image.astype(np.float32), bg_image, label

# 将字符图片的底色与背景图片的底色进行混合
class MixBackgroundColor(object):
    def __call__(self, images, bg_image, label):
        if bg_image is None:
            return images, bg_image, label

        mix_image_lists = []

        alpha_color = [np.random.randint(20,120), np.random.randint(20,120), np.random.randint(20,120)]

        for idx, item in enumerate(images):
            image = item.copy()
            # print('image shape:', image.shape)
            if len(image.shape) == 3:
                # 字符截图，没有做二值化图片
                mix_image_lists.append(self.__normal_mix__(image,bg_image))
            elif len(image.shape) == 2:
                # 字符截图，已做二值化处理图片, 处理完后， image 类型从灰度转换成为了Color
                mix_image_lists.append(self.__custom_mix__(image,bg_image,alpha_color))
            else:
                raise Exception('不能识别的字符串图片类型')

        return mix_image_lists, bg_image, label


    def __normal_mix__(self, image, bg_image):
        if np.random.randint(3) > 0:
            mix_image = np.ones(image.shape, image.dtype) 
            mix_image[:,:,0] = np.median(bg_image[:,:,0])
            mix_image[:,:,1] = np.median(bg_image[:,:,1])
            mix_image[:,:,2] = np.median(bg_image[:,:,2])
            mask = np.ones(image.shape, np.uint8) * 255
            height, width, _ = image.shape
            center_y =  int(height/2)
            center_x =  int(width/2)
            center = (center_x,center_y)
            image = image.astype(np.uint8)
            bg_image = bg_image.astype(np.uint8)
            image = cv2.seamlessClone(image,mix_image, mask,center, cv2.MONOCHROME_TRANSFER)
            return image
        else:
            return image

    def __custom_mix__(self,image,bg_image,alpha_color):
        mix_image = np.ones((image.shape[0], image.shape[1], 3), np.uint8) 
        mix_image[:,:,0] = np.median(bg_image[:,:,0])
        mix_image[:,:,1] = np.median(bg_image[:,:,1])
        mix_image[:,:,2] = np.median(bg_image[:,:,2])
        y, x = np.where(image != 0)
        mix_image[y,x,:] = alpha_color
        return mix_image


# 随机选择字符图片，并在其上标记涂改样式
class RandomMixCleanFlag(object):
    def __init__(self, data_dir, min_radio=0.7, max_radio=0.9, window=300):
        self.data_dir = data_dir
        self.clean_img_lists = self.__load_clean_image__()
        self.min_radio = min_radio
        self.max_radio = max_radio
        self.window = window

    def __load_clean_image__(self):
        image_file_list = os.listdir(self.data_dir)
        image_list = []
        for file_name in image_file_list:
            image = cv2.imread(os.path.sep.join([self.data_dir,file_name]),cv2.IMREAD_COLOR)
            image_list.append(image)
        return image_list

    def __call__(self, images, bg_image, label):
        if bg_image is None:
            return images, bg_image, label

        # print('label:', label, ':', type(label))
        if np.random.randint(10) == 0:
            label_list = list(label)
            image_idx = np.random.randint(len(images))
            clean_img = self.clean_img_lists[np.random.randint(len(self.clean_img_lists))]
            clean_img, scale = self.__adjust_image_size__(clean_img, images[image_idx].copy())
            alpha_color = [np.random.randint(20,120), np.random.randint(20,120), np.random.randint(20,120)]
            images[image_idx] = self.__mix__(clean_img,images[image_idx],alpha_color)
            label_list[image_idx]=BLANK_FLAG  # 该字符标记为空
            label = ''.join(label_list)
        return images, bg_image, label

    # 调整图片大小
    def __adjust_image_size__(self, image,bg_image):
        b_height, b_width,_ = bg_image.shape
        prob = (image.shape[0] * image.shape[1]) / (b_height * b_width)
        pred_prob = np.random.uniform(self.min_radio, self.max_radio)
        scale = min(np.sqrt(pred_prob/prob), b_height/image.shape[0], b_width/image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, scale        

    # 是否根据box 来进行涂改，针对原图的话，box_flag = True, 对于拼接的，则根 box_flag 为False
    def __mix__(self,image,bg_image,alpha_color):
        '''
            image 涂改的图片
            bg_image 需涂改的图片
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height,width,_ = bg_image.shape
        y, x = np.where(image != 0)
        offset_y = np.random.randint((height - np.max(y))) #int((height/2) - np.mean(y)) -1
        offset_x = np.random.randint((width - np.max(x))) #int((width/2) - np.mean(x)) -1
        y = y + offset_y
        x = x + offset_x
        bg_image[y,x,:] = alpha_color
        return bg_image        



class CombinCharImages(object):
    def __init__(self, mean=None):
        self.mean = mean
    def __call__(self, image, bg_image, label):
        if bg_image is None:
            return image, bg_image, label


        height = np.max([x.shape[0] for x in image]) + 10
        width = np.sum([x.shape[1] for x in image]) + 10
        cm_image = np.zeros((height,width, image[0].shape[2]), np.uint8) 
        cm_image = cm_image.astype(image[0].dtype)

        cm_image[:,:,0] = np.median(bg_image[:,:,0])
        cm_image[:,:,1] = np.median(bg_image[:,:,1])
        cm_image[:,:,2] = np.median(bg_image[:,:,2])

        
        offset_x = 0
        offset_y = 0
        for idx,img in enumerate(image):
            height, width , _ = img.shape
            if idx == 0:
                offset_x_bias = 0
            else:
                offset_x_bias = np.random.randint(-2, 2)
            offset_y_bias = np.random.randint(0, 5)
            cm_image[offset_y+offset_y_bias:offset_y+height+offset_y_bias, offset_x+offset_x_bias:offset_x+width+offset_x_bias,:] = img
            offset_x = width + offset_x + offset_x_bias
        image = cm_image.copy()
        return image, bg_image, label


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, bg_image=None, label=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, bg_image, label

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, bg_image=None, label=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, bg_image, label

class RandomSize(object):
    def __init__(self, min_radio=0.8, max_radio=1.4):
        self.min_radio = min_radio
        self.max_radio = max_radio


    def __call__(self, image,bg_image=None, label=None):
        # if random.randint(2):
        if np.random.randint(2):
            radio = random.uniform(self.min_radio, self.max_radio)
            height, width, _ = image.shape
            image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_NEAREST)
        return image,bg_image, label


class CharImgTransform(object):
    '''
    将原图与背景图进行混合的处理
    '''
    def __init__(self, data_root, imgH=32):
        self.imgH = imgH
        self.data_root = data_root
        self.augment = Compose([
            MixBackgroundColor(),
            RandomMixCleanFlag(data_dir=os.path.sep.join([self.data_root, 'dest', 'CleanOrigin'])),
            CombinCharImages(),     # 合并有多个字符串的截图
            RandomSize(),
            ConvertFromInts(),      # 检测图像是否为BGR，如是则转成灰度模式
            RandomBrightness()
        ])

    def __call__(self, char_img, bg_image, label):
        return self.augment(char_img, bg_image, label)

