import torch
from torchvision import transforms
import cv2
import numpy as np
from numpy import random
from math import fabs,sin,radians,cos
from os.path import join
import os


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    img = gray.copy()
    blur = cv2.GaussianBlur(img, (7,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)    
    return dilate

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
    x0 = max(np.min(box_lists[:,0]) ,0)
    y0 = max(np.min(box_lists[:,1]) ,0)
    x1 = min(np.max(box_lists[:,2]) , width)
    y1 = min(np.max(box_lists[:,3]) , height)

    box = (x0,y0,x1,y1)

    return box

# 旋转图片
def rotate_image(image, degree, border_value=(255,255,255)):
    height,width=image.shape[:2]
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2  #重点在这步，目前不懂为什么加这步
    matRotation[1,2] +=(heightNew-height)/2  #重点在这步
    imgRotation=cv2.warpAffine(image,matRotation,(widthNew,heightNew),borderValue=border_value)

    return imgRotation, matRotation


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, bg_img):
        for t in self.transforms:
            img, label,bg_img = t(img, label,bg_img)
        return img, label, bg_img

class ConvertFromInts(object):
    def __call__(self, image, label, bg_img):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32), label, bg_img

# 去文字白边
class RemoveWhiteBoard(object):
    def __call__(self, image, label, bg_img):
        if label == 'bz':
            return image, label, bg_img

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = preprocess(gray)

        # 3. 查找和筛选文字区域
        text_boxes = findTextRegion(dilation)
        if text_boxes is not None:
            x0,y0,x1, y1 = text_boxes
            rimage = image[y0:y1, x0:x1]
            return rimage, label, bg_img
        else:
            return image, label, bg_img


# 随机裁剪背景图片
class RandomClipBg(object):
    def __init__(self, min_radio=0.5):
        self.min_radio = min_radio

    def __call__(self, image, label, bg_img):
        random_sel = np.random.randint(6)
        if random_sel in [0,1]:
            return image, label, bg_img

        height, width, _ = bg_img.shape
        clip_width = np.random.randint(int(self.min_radio*width), int(0.8 * width))
        # clip_heigth = np.random.randint(int(self.min_radio*height), int(0.8 * height))
        clip_bg_img = np.ones((height, clip_width, 3), dtype=bg_img.dtype) * 255

        if random_sel == 2:
            clip_bg_img[0:width, 0:clip_width,] = bg_img[0:height, 0:clip_width, ]
        elif random_sel == 3:
            clip_bg_img[0:width, 0:clip_width,] = bg_img[0:height, width-clip_width:width, ]
        else:
            clip_width_center = clip_width // 2
            clip_bg_img[0:height, 0:clip_width_center, ] = bg_img[0:height, 0:clip_width_center,] 
            clip_bg_img[0:height, clip_width_center:clip_width, ] = bg_img[0:height, width-(clip_width-clip_width_center):width, ]
        return image, label, clip_bg_img





# # 随机扩展图片高度、宽度
class RandomExpandBg(object):
    def __init__(self, e_max_height=0.3, e_max_width=0.2):
        self.e_max_width = e_max_width
        self.e_max_height = e_max_height

    def __call__(self, image, label, bg_img):
        random_sel = np.random.randint(6)
        if random_sel in [0,1]:
            return image, label, bg_img

        height, width, _ = bg_img.shape
        exp_width = np.random.randint(1, int(self.e_max_width * width)) + width
        exp_height = np.random.randint(1, int(self.e_max_height * height)) + height
        image_ext = np.ones((exp_height, exp_width, 3),dtype=image.dtype) * 255
        image_ext[:,0] = np.mean(bg_img[:,:,0])
        image_ext[:,1] = np.mean(bg_img[:,:,1])
        image_ext[:,2] = np.mean(bg_img[:,:,2])

        x_pos = np.random.randint(0, exp_width-width)
        y_pos = np.random.randint(0, exp_height-height)
        image_ext[y_pos:y_pos+height, x_pos:x_pos+width,:] = bg_img
        return image, label, image_ext


# 随机增加噪点
class RandomNoise(object):
    def __init__(self, sp_prob=0.005, gas_mean=0, gas_var=0.001):
        self.sp_prob = sp_prob
        self.gas_mean = gas_mean
        self.gas_var = gas_var

    def __sp_noise__(self,image,prob):
        '''
        添加椒盐噪声
        prob:噪声比例
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def __gasuss_noise__(self, image, mean=0, var=0.001):

        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out    


    def __call__(self, image, label, bg_img):
        random_sel = np.random.randint(4)            
        if random_sel == 0:
            image = self.__sp_noise__(image, self.sp_prob)
        elif random_sel in [1,2]:
            image = self.__gasuss_noise__(image, self.gas_mean, self.gas_var)
        return image, label, bg_img


# 增加背景图片或将图片直接转成背景图片，但需要将labels同时替换成z符号
class RandomBackGround(object):
    def __init__(self, alpha_min=0.3):
        self.alpha_min = alpha_min


    def __mask_img__(self, bg_img, image):
        '''
        在背景图上面粘贴二维码识别图片
        is_origin 表示是否是原始图片
        '''
        # x_pos, y_pos = mask_pos
        
        bg_height, bg_width, _ = bg_img.shape
        height_radio = np.random.uniform((bg_height-1)/image.shape[0] * 0.5 ,(bg_height-1)/image.shape[0] * 0.9)
        width_radio = np.random.uniform((bg_width-3)/image.shape[1] * 0.3 ,(bg_width-3)/image.shape[1] * 0.7)
        image = cv2.resize(image.copy(),(0,0), fx=width_radio,fy=height_radio, interpolation=cv2.INTER_AREA)

        _bg_img = bg_img.copy()

        x_pos, y_pos = np.random.randint(3, bg_width - image.shape[1] - 3),np.random.randint(1, bg_height - image.shape[0] - 1)
        mix_image = np.ones(bg_img.shape, bg_img.dtype) * 255
        mix_image[y_pos:y_pos+image.shape[0], x_pos:x_pos+image.shape[1], :] = image        

        # mix_area_img = bg_img[y_pos:y_pos+image.shape[0], x_pos:x_pos+image.shape[1],:]
        # mix_image[:,:,0] = np.median(mix_area_img[:,:,0])
        # mix_image[:,:,1] = np.median(mix_area_img[:,:,1])
        # mix_image[:,:,2] = np.median(mix_area_img[:,:,2])


        alpha_min = self.alpha_min
        alpha_1 = np.random.uniform(alpha_min, 0.7)
        # alpha_1 = 0.2
        mix_image = cv2.addWeighted(mix_image, alpha_1, _bg_img, 1-alpha_1, 0)

        # _bg_img[y_pos:y_pos+image.shape[0], x_pos:x_pos+image.shape[1], :] = _img
        return mix_image

    def __call__(self, image, label, bg_img):
        if label == 'bz':
            return image, label, bg_img

        image = self.__mask_img__(bg_img, image)
        return image, label, bg_img

class RandomBlur(object):
    def __init__(self):
        self.ksize = [3,1]

    def mean_blur(self,image, a, b):
        # (a,b)表示的卷积核[大小]  a代表横向上的模糊，b表示纵向上的模糊
        dst = cv2.blur(image, (a, b))
        return dst

    def median_blur(self, image, ksize):
        # 第二个参数是孔径的尺寸，一个大于1的奇数。
        # 比如这里是5，中值滤波器就会使用5×5的范围来计算。
        # 即对像素的中心值及其5×5邻域组成了一个数值集，对其进行处理计算，当前像素被其中值替换掉。
        dst = cv2.medianBlur(image, ksize)
        return dst

    def __call__(self, image, label, bg_img):
        r_idx = np.random.randint(4)
        # print('----------------------------------blur idx :', r_idx)
        if r_idx in [0,1]:
            image = self.mean_blur(image, np.random.randint(1,6), np.random.randint(1,6))
        # elif r_idx in [2,3]:
        # image = self.median_blur(image, self.ksize[np.random.randint(len(self.ksize))])

        return image, label, bg_img

# 随机旋转图片
class RandomRotate(object):
    # 随机旋转QR图片
    def __init__(self, degree_range=(-5,5)):
        self.degree_range = degree_range

    def __call__(self, img, label,bg_img):
        random_sel = np.random.randint(3)
        if random_sel == 0:
            degree = np.random.uniform(self.degree_range[0], self.degree_range[1])
            border_value = (255,255,255)
            _img, _ = rotate_image(img, degree,border_value)
            return _img, label, bg_img
        else:
            return img, label, bg_img

# 随机改变大小
class RandomSize(object):
    def __init__(self, min_radio=0.8, max_radio=1.2, min_size=900, min_height=26):
        self.min_radio = min_radio
        self.max_radio = max_radio
        self.min_height = min_height
        self.min_size = 900
    def __call__(self, image, label, bg_img):
        height, width, _ = image.shape
        min_radio = self.min_radio
        max_radio = self.max_radio
        # radio = random.uniform(min_radio, max_radio)
        radio = 0.2
        radio = max(radio, self.min_height/height)
        random_sel = np.random.randint(4)
        if random_sel == 0:
            image = cv2.resize(image.copy(),(0,0), fx=radio,fy=radio, interpolation=cv2.INTER_AREA)
        elif random_sel == 1:
            image = cv2.resize(image.copy(),(0,0), fx=radio,fy=radio*1.2, interpolation=cv2.INTER_AREA)
        elif random_sel == 2:
            image = cv2.resize(image.copy(),(0,0), fx=radio*1.2,fy=radio, interpolation=cv2.INTER_AREA)
        return image,label, bg_img





class ImgTransform(object):
    def __init__(self, data_root):
        self.data_root = data_root

        self.augment = Compose([
                # SelectEmptyImage(data_root=self.data_root),
                RandomClipBg(),
                RandomExpandBg(),
                RemoveWhiteBoard(),
                RandomBackGround(),
                RandomRotate(),
                RandomSize(),
                RandomNoise(),
                RandomBlur(),
                ConvertFromInts()
            ])


    def __call__(self, image, label, bg_img):
        return self.augment(image, label, bg_img)
