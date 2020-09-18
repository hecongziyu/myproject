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

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels

class ConvertFromInts(object):
    def __call__(self, image, labels):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32), labels

# 去文字白边
class RemoveWhiteBoard(object):
    def __call__(self, image, labels):
        if labels == 'z':
            return image, labels
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 形态学变换的预处理，得到可以查找矩形的图片
        dilation = preprocess(gray)

        # 3. 查找和筛选文字区域
        text_boxes = findTextRegion(dilation)
        if text_boxes is not None:
            x0,y0,x1, y1 = text_boxes
            rimage = image[y0:y1, x0:x1]
            return rimage, labels
        else:
            return image, labels

# 随机改变大小
class RandomSize(object):
    def __init__(self, min_radio=0.4, max_radio=1, min_size=900, min_height=32):
        self.min_radio = min_radio
        self.max_radio = max_radio
        self.min_height = min_height
        self.min_size = 900
    def __call__(self, image, labels):
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
            image = cv2.resize(image.copy(),(0,0), fx=radio,fy=radio*1.5, interpolation=cv2.INTER_AREA)
        elif random_sel == 2:
            image = cv2.resize(image.copy(),(0,0), fx=radio*1.5,fy=radio, interpolation=cv2.INTER_AREA)
        return image,labels


# 为label为z的图片选择图片
class SelectEmptyImage(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.use_img_lists = None

    def __load__use_img__(self):
        _file_list = os.listdir(join(self.data_root, 'useimg'))
        img_lists = []
        for item in _file_list:
            image = cv2.imread(join(self.data_root, 'useimg', item), cv2.IMREAD_COLOR)
            img_lists.append(image)
        return img_lists

    def __call__(self, image, labels):
        if self.use_img_lists is None:
            self.use_img_lists = self.__load__use_img__()

        if labels == 'z' and image is None:
            _image = self.use_img_lists[np.random.randint(len(self.use_img_lists))]

            # 将图片随机做翻转，后面实现 

            height, width, _ = _image.shape
            x_pos = np.random.randint(0, width - 20)
            y_pos = np.random.randint(0, height - 20)

            x_width = np.random.randint(50,300)
            x_height = np.random.randint(50,128)

            image = _image[y_pos:min(y_pos+x_height, height), x_pos:min(x_pos+x_width, width), ]

        return image, labels

# 随机旋转图片
class RandomRotate(object):
    # 随机旋转QR图片
    def __init__(self, degree_range=(-10,10)):
        self.degree_range = degree_range

    def __call__(self, img, labels):
        if np.random.randint(2) == 0:
            degree = np.random.uniform(self.degree_range[0], self.degree_range[1])
            border_value = (255,255,255)
            img, _ = rotate_image(img, degree,border_value)
        return img, labels

# 增加背景图片或将图片直接转成背景图片，但需要将labels同时替换成z符号
class RandomBackGround(object):
    def __init__(self, data_root, alpha_min=0.3):
        self.data_root = data_root
        self.bg_img_lists = None
        self.alpha_min = alpha_min

    def __load__bg_img__(self):
        _file_list = os.listdir(join(self.data_root, 'bgimg'))
        img_lists = []
        for item in _file_list:
            image = cv2.imread(join(self.data_root, 'bgimg', item), cv2.IMREAD_COLOR)
            img_lists.append(image)
        return img_lists

    def __mask_img__(self, bg_img, image):
        '''
        在背景图上面粘贴二维码识别图片
        is_origin 表示是否是原始图片
        '''
        # x_pos, y_pos = mask_pos
        height, width, _ = image.shape
        bg_img = cv2.resize(bg_img.copy(), (width,height), interpolation=cv2.INTER_AREA)
        alpha_min = self.alpha_min
        alpha_1 = np.random.uniform(alpha_min, 0.95)
        _img = cv2.addWeighted(image, alpha_1, bg_img, 1-alpha_1, 0)
        return _img


    def __call__(self, image, labels):
        if self.bg_img_lists is None:
            self.bg_img_lists = self.__load__bg_img__()

        if len(self.bg_img_lists) == 0:
            return image, labels

        if labels == 'z':
            return image, labels

        if np.random.randint(5) in [1,2,0]:
            bg_img = self.bg_img_lists[np.random.randint(len(self.bg_img_lists))]
            image = self.__mask_img__(bg_img, image)
        else:
            if labels != 'z':
                if np.random.randint(2):
                    image = self.bg_img_lists[np.random.randint(len(self.bg_img_lists))]
                    labels = 'z' 
        
        return image, labels

# 随机增加噪点
class RandomNoise(object):
    def __init__(self, sp_prob=0.015, gas_mean=0, gas_var=0.001):
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


    def __call__(self, image, labels):
        random_sel = np.random.randint(4)            
        if random_sel == 0:
            image = self.__sp_noise__(image, self.sp_prob)
        elif random_sel in [1,2]:
            image = self.__gasuss_noise__(image, self.gas_mean, self.gas_var)
        return image, labels

# # 随机扩展图片高度、宽度
class RandomExpand(object):
    def __init__(self, e_max_height=20, e_max_width=60):
        self.e_max_width = e_max_width
        self.e_max_height = e_max_height

    def __call__(self, image, labels):
        if labels == 'z':
            return image, labels

        random_sel = np.random.randint(6)
        if random_sel in [0,1]:
            return image, labels

        height, width, _ = image.shape
        exp_width = np.random.randint(1, self.e_max_width) + width
        exp_height = np.random.randint(1,self.e_max_height) + height
        image_ext = np.ones((exp_height, exp_width, image.shape[2]),dtype=image.dtype) * 255
        x_pos = np.random.randint(0, exp_width-width)
        y_pos = np.random.randint(0, exp_height-height)
        image_ext[y_pos:y_pos+height, x_pos:x_pos+width,:] = image
        return image_ext, labels


class RandomBlur(object):
    def __init__(self):
        self.ksize = [3,5]

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

    def __call__(self, image, labels):
        r_idx = np.random.randint(5)
        # print('----------------------------------blur idx :', r_idx)
        if r_idx in [0,1]:
            image = self.mean_blur(image, np.random.randint(1,6), np.random.randint(1,6))
        elif r_idx in [2,3]:
            image = self.median_blur(image, self.ksize[np.random.randint(len(self.ksize))])

        return image, labels



class ImgTransform(object):
    def __init__(self, data_root):
        self.data_root = data_root

        self.augment = Compose([
                SelectEmptyImage(data_root=self.data_root),
                RemoveWhiteBoard(),
                RandomSize(),
                RandomExpand(),
                RandomRotate(),
                RandomBackGround(data_root=self.data_root),
                RandomNoise(),
                RandomBlur(),
                ConvertFromInts()
            ])


    def __call__(self, image, labels):
        return self.augment(image, labels)

# class ImgTransform(object):
#     def __init__(self, data_root):
#         self.data_root = data_root

#         self.augment = Compose([
#                 SelectEmptyImage(data_root=self.data_root),
#                 RemoveWhiteBoard(),
#                 RandomExpand(),
#                 RandomSize(),
#                 RandomNoise(),
#                 RandomRotate(),
#                 RandomBackGround(data_root=self.data_root),
#                 # ConvertFromInts()
#             ])


#     def __call__(self, image, labels):
#         return self.augment(image, labels)


