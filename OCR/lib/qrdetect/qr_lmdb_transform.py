import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import fabs,sin,radians,cos

# https://www.cnblogs.com/lfri/p/10627595.html 图片随机添加噪声
# 
# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] - 15 or
                first[0] > other[2] + 15 or
                first[1] > other[3] + 15 or
                first[3] < other[1] - 15 )


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, qr_img, boxes=None):
        for t in self.transforms:
            img, qr_img, boxes = t(img, qr_img, boxes)
        return img, qr_img, boxes


def randomQrSize(image, min_height=24, min_width=24, is_origin=False):
    height, width, _ = image.shape
    if is_origin:
        radio = np.random.uniform(0.4, 0.60)
    else:
        radio = np.random.uniform(0.7, 1.2)

    radio = max(radio, min_height/height, min_width/width)

    random_sel = np.random.randint(5)

    if random_sel in [0,1]:
        image = cv2.resize(image, (0,0), fx=radio,fy=radio,interpolation=cv2.INTER_AREA)
    elif random_sel == 2:
        image = cv2.resize(image, (0,0), fx=radio,fy=radio*1.2,interpolation=cv2.INTER_AREA)
    elif random_sel == 3:
        image = cv2.resize(image, (0,0), fx=radio*1.2,fy=radio,interpolation=cv2.INTER_AREA)

    return image


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
    



# 随机粘贴二维码图片到底片上面
# https://blog.csdn.net/fanjiule/article/details/81607873 opencv中addWeighted()函数用法总结
class MaskQRImage(object):

    def __init__(self, qr_alpha_min=0.4):
        '''
        qr_alpha : 在粘贴时前景图片权重
        qr_alpha + bg_alpha = 1.0

        注意： 因为需判断是否是原始QR图片(原始图片大小为 70*70大小格式)
        所以对QR图片的缩放、变形都放在该类里面进行处理
        '''
        self.qr_alpha_min = qr_alpha_min


    def __mask_img__(self, bg_img, image, mask_pos, is_origin=False):
        '''
        在背景图上面粘贴二维码识别图片
        is_origin 表示是否是原始图片
        '''
        x_pos, y_pos = mask_pos
        q_height, q_width, _ = qr_img.shape

        mix_image = np.ones(qr_img.shape, qr_img.dtype) 
        mix_area_img = bg_img[y_pos-10:y_pos+qr_img.shape[0]+10, x_pos-10:x_pos+qr_img.shape[1]+10,:]
        mix_image[:,:,0] = np.median(mix_area_img[:,:,0])
        mix_image[:,:,1] = np.median(mix_area_img[:,:,1])
        mix_image[:,:,2] = np.median(mix_area_img[:,:,2])

        qr_alpha_min = self.qr_alpha_min

        if not is_origin:
            qr_alpha_min = 0.9

        alpha_1 = np.random.uniform(qr_alpha_min, 0.95)

        _img = cv2.addWeighted(qr_img, alpha_1, mix_image, 1-alpha_1, 0)

        # print('bg img :', bg_img.shape, ' qr img shape :', _img.shape, ' x pos :', x_pos, ' y pos :', y_pos)
        # print('y area :', y_pos, '----',y_pos+q_height)
        # print('x area :', x_pos, '----',x_pos+q_width)
        bg_img[y_pos:y_pos+q_height, x_pos:x_pos+q_width, :] = _img
        return bg_img





    def __call__(self, image, labels):
        height, width, _ = image.shape

        if np.random.randint(4) in [0,1,2]:
            

        return image,labels


# https://blog.csdn.net/qq_38395705/article/details/106311905?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param
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


    def __call__(self, img, qr_img, boxes):
        random_sel = np.random.randint(5)            
        if random_sel == 0:
            img = self.__sp_noise__(img, self.sp_prob)
        elif random_sel == 1:
            qr_img = self.__sp_noise__(qr_img, self.sp_prob)
        elif random_sel == 2:
            img = self.__gasuss_noise__(img, self.gas_mean, self.gas_var)
        elif random_sel == 3:
            qr_img = self.__gasuss_noise__(qr_img, self.gas_mean, self.gas_var)
        return img, qr_img, boxes

class RandomRemoveQR(object):
    # 随机移走0到2个QR
    def __init__(self, number=2):
        self.number =  number

    def __call__(self, img, qr_img, boxes):
        random_sel = np.random.randint(0, boxes.shape[0], np.random.randint(self.number))
        _tmp_boxes = boxes.copy()
        # print('del boxes :', boxes.shape, 'random sel :', random_sel)
        for idx in random_sel:
            x0,y0,x1,y1,label = _tmp_boxes[idx]
            if label != -1 :
                img[y0:y1,x0:x1, 0] = np.median(img[:,:,0])
                img[y0:y1,x0:x1, 1] = np.median(img[:,:,1])
                img[y0:y1,x0:x1, 2] = np.median(img[:,:,2])
                boxes = np.delete(boxes, idx, axis=0)

        if boxes.shape[0] == 0:
            boxes = np.array([[-1,-1,-1,-1,-1]])

        return img, qr_img, boxes

class RandomRotateQR(object):
    # 随机旋转QR图片
    def __init__(self, degree_range=(-20,20)):
        self.degree_range = degree_range

    def __call__(self, img, qr_img, boxes):
        if np.random.randint(2) == 0:
            degree = np.random.uniform(self.degree_range[0], self.degree_range[1])
            border_value = (np.median(img[:,:,0]),np.median(img[:,:,1]),np.median(img[:,:,2]))
            qr_img, _ = rotate_image(qr_img, degree,border_value)
        return img, qr_img, boxes


class RandomRotateImg(object):
    # 随机旋转整个图片，并调整boxes对应的位置信息
    def __init__(self):
        self.normal_degree = [0,90,180,-90]
        self.degree_range = (-15,15)

    def get_box_pos(self,src_pos, map_rotation):
        x0,y0,x1,y1 = src_pos
        _src_pos = [(x0,y0),(x0,y1),(x1,y0),(x1,y1)]
        _pos_x_list = []
        _pos_y_list = []
        for pos in _src_pos:
            x,y = pos
            _x,_y = np.dot(map_rotation, np.array([[x],[y],[1]]))
            _pos_x_list.append(int(_x))
            _pos_y_list.append(int(_y))
        dest_pos = [min(_pos_x_list), min(_pos_y_list), max(_pos_x_list), max(_pos_y_list)]
        return dest_pos    

    def __call__(self, img, qr_img, boxes):
        random_sel = np.random.randint(4)
        degree = 0
        if random_sel == 0:
            degree = self.normal_degree[np.random.randint(len(self.normal_degree))]
        elif random_sel == 1:
            degree = np.random.uniform(self.degree_range[0], self.degree_range[1])
        elif random_sel == 2:
            degree = np.random.uniform(-5, 5)
        else:
            pass

        if degree != 0:
            border_value = (np.median(img[:,:,0]),np.median(img[:,:,1]),np.median(img[:,:,2]))
            img, map_rotataion = rotate_image(img, degree,border_value)
            boxes_lists = []
            for box in boxes.tolist():
                label = box[-1]
                x0,y0,x1,y1 = self.get_box_pos(box[0:4], map_rotataion)
                boxes_lists.append([x0,y0,x1,y1,label])
            boxes = np.array(boxes_lists)

        return img, qr_img, boxes



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
            RandomNoise(sp_prob=0.005, gas_mean=0, gas_var=0.0005),
            RandomRotateQR(),
            RandomRotateImg(),
            MaskQRImage(),
            RandomRemoveQR(),
            AdjustSize(),
            Mask2Windows(window=self.window),
            StructResize(size=self.size),
            ConvertFromInts()

        ])

    def __call__(self, img, qr_img, boxes):
        return self.augment(img, qr_img, boxes)    



# 测试用的转换器

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
