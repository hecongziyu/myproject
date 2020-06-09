import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from matplotlib import pyplot as plt
from gtdb import box_utils
from PIL import Image, ImageOps
import os


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, bg_img=None):
        for t in self.transforms:
            img, boxes, bg_img = t(img, boxes, bg_img)
        return img, boxes, bg_img


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, bg_img=None):
        return image.astype(np.float32), boxes, bg_img

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, bg_img=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, bg_img

class ToAbsoluteCoords(object):
    def __init__(self, window):
        self.window = window

    def __call__(self, image, boxes=None, bg_img=None):
        # height, width, channels = image.shape
        boxes[:, 0] *= self.window
        boxes[:, 2] *= self.window
        boxes[:, 1] *= self.window
        boxes[:, 3] *= self.window
        return image, boxes, bg_img

class ToPercentCoords(object):
    def __init__(self, window):
        self.window = window
    def __call__(self, image, boxes=None, bg_img=None):
        height, width, channels = image.shape
        boxes[np.where(boxes[:,4] == -1),0:4] = -1
        boxes[:, 0] /= self.window
        boxes[:, 2] /= self.window
        boxes[:, 1] /= self.window
        boxes[:, 3] /= self.window

        return image, boxes, bg_img


class ExpandBackGround(object):
    def __init__(self, min_width=60, min_height=32):
        self.min_width = min_width
        self.min_height = min_height

    def __call__(self, images, boxes, bg_image):
        if bg_image is None:
            return images, boxes, bg_image

        height, width, channel = bg_image.shape
        if bg_image.shape[0] < self.min_height or bg_image.shape[1] < self.min_width:
            # print('expand image ', bg_image.shape, ' mean :', np.mean(bg_image[:,:,0]),":",np.mean(bg_image[:,:,1]))
            exp_img = np.zeros((max(self.min_height,bg_image.shape[0]) + 1, 
                                max(self.min_width, bg_image.shape[1]) + 1, bg_image.shape[2])) 
            exp_img[:,:,0] = np.median(bg_image[:,:,0])
            exp_img[:,:,1] = np.median(bg_image[:,:,1])
            exp_img[:,:,2] = np.median(bg_image[:,:,2])

            exp_img = exp_img.astype(np.uint8)
            e_height, e_width, _ = exp_img.shape

            e_h_pos = int((e_height - height)/2)
            e_w_pos = int((e_width - width)/2)
            # exp_img[0:height, 0:width,:] = bg_image.copy()

            # 填充前半部分
            exp_img[e_h_pos:e_h_pos+height,0:int(width/2),:] = bg_image[0:height, 0:int(width/2),:]
            # 填充后半部分
            exp_img[e_h_pos:e_h_pos+height,e_width-int(width/2):e_width,:] = bg_image[0:height, width - int(width/2):width,:]
            bg_image = exp_img.copy()

            scale = np.random.uniform(1.0,1.3)
            bg_image = cv2.resize(bg_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        return images, boxes, bg_image


class Mask2Windows(object):
    def __init__(self, window):
        self.window = window

    def __call__(self, image, boxes=None,bg_img=None):
        height, width, _ = image.shape
        if height > self.window or width > self.window:
            return image, boxes, bg_img

        win_img = np.full((self.window, self.window, image.shape[2]), 255)
        win_img[0:image.shape[0],0:image.shape[1],:] = image.copy()
        return win_img, boxes, bg_img

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, bg_img=None):
        # print('resize img shape:', image.shape)
        image = cv2.resize(image.copy(), (self.size,self.size), interpolation=cv2.INTER_NEAREST)
        return image, boxes, bg_img

class RandomSize(object):
    def __init__(self, min_radio=0.9, max_radio=1.5):
        self.min_radio = min_radio
        self.max_radio = max_radio


    def __call__(self, image,boxes=None, bg_img=None):
        # if random.randint(2):
        if np.random.randint(2):
            radio = random.uniform(self.min_radio, self.max_radio)
            height, width, _ = image.shape
            if width * radio <= 300:
                image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_NEAREST)
                boxes[:,0] = boxes[:,0] * radio
                boxes[:,1] = boxes[:,1] * radio
                boxes[:,2] = boxes[:,2] * radio
                boxes[:,3] = boxes[:,3] * radio
        return image,boxes, bg_img


# 将字符图片的底色与背景图片的底色进行混合
class MixBackgroundColor(object):
    def __call__(self, images, boxes, bg_image):
        if bg_image is None:
            return images, boxes, bg_image

        mix_image_lists = []

        alpha_color = [np.random.randint(20,120), np.random.randint(20,120), np.random.randint(20,120)]

        for idx, item in enumerate(images):
            image, img_type = item
            if img_type == 0:
                # 字符截图，没有做二值化图片
                mix_image_lists.append(self.__normal_mix__(image,bg_image))
            elif img_type == 1:
                # 字符截图，已做二值化处理图片
                mix_image_lists.append(self.__custom_mix__(image,bg_image, alpha_color))
            else:
                raise Exception('不能识别的字符串图片类型')

        return mix_image_lists, boxes, bg_image


    def __normal_mix__(self, image, bg_image):
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
        # MONOCHROME_TRANSFER
        image = cv2.seamlessClone(image,mix_image, mask,center, cv2.MONOCHROME_TRANSFER)
        return image

    def __custom_mix__(self,image,bg_image,alpha_color):
        mix_image = np.ones(image.shape, image.dtype) * np.median(bg_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def __call__(self, images, boxes, bg_image):
        # if bg_image is None:
        #     return images, boxes, bg_image

        if np.random.randint(5) == 0:
            image_idx = np.random.randint(len(images))
            if boxes[image_idx, 4] != -1:
                clean_img = self.clean_img_lists[np.random.randint(len(self.clean_img_lists))]
                box_flag = True if bg_image is None else False
                clean_img, scale = self.__adjust_image_size__(clean_img, images[image_idx].copy(),boxes[image_idx].copy(), box_flag)
                alpha_color = [np.random.randint(20,120), np.random.randint(20,120), np.random.randint(20,120)]
                images[image_idx] = self.__mix__(clean_img,images[image_idx],boxes[image_idx].copy(),alpha_color,box_flag)
                if len(images) == 1:
                    boxes[image_idx] = [-1,-1,-1,-1,-1]

        return images, boxes, bg_image


    # 调整图片大小
    def __adjust_image_size__(self, image,bg_image, box, box_flag=False):
        box = box * self.window
        box = box.astype(np.uint8)
        if box_flag:
            b_height = int(box[3] - box[1])
            b_width = int(box[2] - box[0])
        else:
            b_height, b_width,_ = bg_image.shape
        prob = (image.shape[0] * image.shape[1]) / (b_height * b_width)
        pred_prob = np.random.uniform(self.min_radio, self.max_radio)
        scale = min(np.sqrt(pred_prob/prob), b_height/image.shape[0], b_width/image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        # print('scale :', scale)
        return image, scale        

    # 是否根据box 来进行涂改，针对原图的话，box_flag = True, 对于拼接的，则根 box_flag 为False
    def __mix__(self,image,bg_image,box,alpha_color,box_flag=False):
        '''
            image 涂改的图片
            bg_image 需涂改的图片
        '''
        box = box * self.window
        box = box.astype(np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height,width,_ = bg_image.shape
        y, x = np.where(image != 0)
        if box_flag:
            offset_y = box[1] + np.random.randint(box[3] - box[1] - np.max(y))
            offset_x = box[0] + np.random.randint(box[2] - box[0] - np.max(x))
        else:
            offset_y = np.random.randint((height - np.max(y))) #int((height/2) - np.mean(y)) -1
            offset_x = np.random.randint((width - np.max(x))) #int((width/2) - np.mean(x)) -1

        y = y + offset_y
        x = x + offset_x
        bg_image[y,x,:] = alpha_color
        return bg_image        



# 字符图片混合, 注意该混合是以水平直接混合，后期需修改为垂直、水平混合合成
# 注意返回的时候，图片为合并后的单张图片, boxes 可以是在区域中的多个位置信息
class CombinCharImages(object):
    def __init__(self, window, mean=None):
        self.mean = mean
        self.window = window

    def __call__(self, images, boxes, bg_image):
        if bg_image is None or type(images) != list or len(images) == 1:
            return images[0], boxes, bg_image

        image_lists = []
        boxes_lists = []

        # 把多图片进行合并，按左右顺序， 随机分隔成两行
        split_images_lists, split_boxes_lists = self.__random_split_data__(images, boxes)
        for idx   in range(len(split_images_lists)):
            split_images = split_images_lists[idx]
            split_boxes = split_boxes_lists[idx]
            split_images, split_boxes = self.__combine_to_lines__(split_images,split_boxes, bg_image)
            image_lists.append(split_images[0])
            boxes_lists.extend(split_boxes)
        boxes_lists = np.array(boxes_lists)
        c_width = max([x.shape[1] for x in image_lists]) + 1
        c_heigh = sum([x.shape[0] for x in image_lists]) + 5
        c_image = np.ones((c_heigh,c_width, 3), np.uint8)
        c_image[:,:,0] = np.median(bg_image[:,:,0])
        c_image[:,:,1] = np.median(bg_image[:,:,1])
        c_image[:,:,2] = np.median(bg_image[:,:,2])

        # 把两行图片进行合并，按上下顺序
        offset_y = 0
        for idx  in range(len(image_lists)):
            m_image = image_lists[idx]
            offset_x = np.random.randint(0, c_width-m_image.shape[1])
            c_image[offset_y:offset_y+m_image.shape[0], offset_x:offset_x + m_image.shape[1],:] = m_image
            boxes_lists[idx,1] = boxes_lists[idx,1] + offset_y
            boxes_lists[idx,3] = boxes_lists[idx,3] + offset_y
            boxes_lists[idx,0] = boxes_lists[idx,0] + offset_x
            boxes_lists[idx,2] = boxes_lists[idx,2] + offset_x
            offset_y = m_image.shape[0] + np.random.randint(1, 4)



        boxes_lists = boxes_lists/self.window
        
        return c_image,boxes_lists,bg_image

    # 随机将images切割成2份
    def __random_split_data__(self, images,boxes):
        split_images = []
        split_boxes = []        
        if np.random.randint(2) and len(images) > 2 :
            split_number = np.random.randint(1, len(images))
            split_images.append(images[0:split_number])
            split_boxes.append(boxes[0:split_number])
            split_images.append(images[split_number:])
            split_boxes.append(boxes[split_number:])
        else:
            split_images.append(images)
            split_boxes.append(boxes)
        return split_images,split_boxes





    def __combine_to_lines__(self, images, split_boxes, bg_image):
        combin_image_lists = []
        combin_boxes_lists = []

        height = np.max([x.shape[0] for x in images])
        width = np.sum([x.shape[1] for x in images])
        cm_image = np.zeros((height,width, images[0].shape[2]))
        cm_image = cm_image.astype(images[0].dtype)

        if self.mean is None:
            cm_image[:,:,0] = np.median(bg_image[:,:,0])
            cm_image[:,:,1] = np.median(bg_image[:,:,1])
            cm_image[:,:,2] = np.median(bg_image[:,:,2])
        else:
            # print('mean:', self.mean)
            cm_image = cm_image * self.mean
        
        offset_x = 0
        offset_y = 0

        for idx,img in enumerate(images):
            height, width , _ = img.shape
            # print('--->', img.shape, ':', offset_y,':', offset_y+height)
            if idx == 0:
                offset_x_bias = 0
            else:
                offset_x_bias = np.random.randint(6)
                # offset_x_bias = 6
            cm_image[offset_y:offset_y+height, offset_x-offset_x_bias:offset_x+width-offset_x_bias,:] = img
            offset_x = width + offset_x - offset_x_bias


        combin_image_lists.append(cm_image)
        combin_boxes_lists.append([0.,0.,cm_image.shape[1], cm_image.shape[0], np.max(split_boxes[:,4])])
        # combin_boxes_lists = np.array(combin_boxes_lists)
        return combin_image_lists, combin_boxes_lists




# 混合字符图片，如有背景图片，则将其与背景图片进行混合
# 进入的是单张图片
class MaskAlphaImage(object):
    def __init__(self, min_radio=0.3, max_radio=0.8):
        self.min_radio = min_radio
        self.max_radio = max_radio

    # 调整图片大小
    def __adjust_image_size__(self, image,bg_image):
        prob = (image.shape[0] * image.shape[1]) / (bg_image.shape[0] * bg_image.shape[1])
        pred_prob = np.random.uniform(self.min_radio, self.max_radio)
        scale = min(np.sqrt(pred_prob/prob), bg_image.shape[0]/image.shape[0], bg_image.shape[1]/image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, scale        


    def __call__(self, image, boxes=None, bg_image=None):
        if bg_image is None:
            return image, boxes, bg_image
        image, scale = self.__adjust_image_size__(image.copy(), bg_image)
        boxes[:,0:4] = boxes[:,0:4] * scale
        mask = np.ones(image.shape, np.uint8) * 255
        height, width, _ = image.shape
        b_height,b_width,_ = bg_image.shape
        center_y = np.random.randint(int(height/2) , int(b_height-(height/2))) if b_height > height else int(height/2)
        center_x = np.random.randint(int(width/2) , int(b_width-(width/2))) if b_width > width else int(width/2)
        center = (center_x,center_y)
        
        offset_y = center_y - int(height/2)
        offset_x = center_x - int(width/2)

        boxes[:,1] = boxes[:,1] + offset_y
        boxes[:,3] = boxes[:,3] + offset_y
        boxes[:,0] = boxes[:,0] + offset_x
        boxes[:,2] = boxes[:,2] + offset_x

        image = image.astype(np.uint8)
        bg_image = bg_image.astype(np.uint8)

        # MONOCHROME_TRANSFER
        image = cv2.seamlessClone(image,bg_image, mask,center, cv2.MONOCHROME_TRANSFER)

        return image, boxes, bg_image



class GTDBTransform(object):
    def __init__(self, data_root, window=1200, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.window = window
        self.data_root = data_root
        self.augment = Compose([
            MixBackgroundColor(), # 将字符串图片与背景图底色进行混合
            RandomMixCleanFlag(data_dir=os.path.sep.join([self.data_root, 'dest', 'CleanOrigin']), window=self.window), # 将字符串随机增加涂改标记
            CombinCharImages(window=self.window),  # 将多个字符串图片进行合并
            ExpandBackGround(),  # 扩展背景图片大小
            ToAbsoluteCoords(window=self.window),  # ToAbsoluteCoords 转成绝对坐标，生成的box进行了缩放
            MaskAlphaImage(), # 混合背景图片
            RandomSize(), # 随机调整大小
            Mask2Windows(window=self.window), # 将图片粘贴到1200*1200窗口大小的坐标上面
            ConvertFromInts(),
            ToPercentCoords(window=self.window),  # 与ToAbsoluteCoords对应，将target 恢复成x1/width, x2/width, y1/height, y2/height
            Resize(self.size)   # 将变换后图片转成 size * size
        ])

    def __call__(self, img, boxes, bg_img):
        return self.augment(img, boxes,bg_img)
