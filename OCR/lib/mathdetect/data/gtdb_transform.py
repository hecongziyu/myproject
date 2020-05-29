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
    def __init__(self, window):
        self.window = window

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= self.window
        boxes[:, 2] *= self.window
        boxes[:, 1] *= self.window
        boxes[:, 3] *= self.window
        return image, boxes, labels

class ToPercentCoords(object):
    def __init__(self, window):
        self.window = window
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        # print('penc image shape:', image.shape)
        boxes[:, 0] /= self.window
        boxes[:, 2] /= self.window
        boxes[:, 1] /= self.window
        boxes[:, 3] /= self.window

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
                     int(left):int(left + width)] = image.copy()
        # image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        # print('out put boxes:', boxes)
        return expand_image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        # print('resize img shape:', image.shape)
        image = cv2.resize(image.copy(), (self.size,self.size), interpolation=cv2.INTER_AREA)
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
        # bg_img[np.where(image<=246)] = np.random.randint(0,16,len(np.where(image<=246)[0]))
        bg_img[np.where(image<=246)] = 0
        return bg_img,boxes,labels

# 将图片贴在设定的窗口里面
class Mask2Windows(object):
    def __init__(self, window):
        self.window = window

    def __call__(self, image, boxes=None, labels=None):
        win_img = np.full((self.window, self.window, image.shape[2]), 255)
        if random.randint(3) == 0:
            win_img[0:image.shape[0],0:image.shape[1],:] = image.copy()
        else:
            # 原图在window上面随机偏移位置
            xl = random.randint(self.window - image.shape[1])
            yl = random.randint(self.window - image.shape[0])
            win_img[yl:yl+image.shape[0], xl:xl+image.shape[1],:] = image.copy()
            boxes = boxes.copy()
            boxes[:,(0,2)] += xl
            boxes[:,(1,3)] += yl
        return win_img, boxes, labels

class RandomSize(object):
    def __init__(self, min_radio=0.8, max_radio=1):
        self.min_radio = min_radio
        self.max_radio = max_radio


    def __call__(self, image,boxes=None, labels=None):
        if random.randint(2):
            radio = random.uniform(self.min_radio, self.max_radio)
            # radio = 0.999
            height, width, _ = image.shape
            image = cv2.resize(image.copy(), (int(width*radio),int(height * radio)), interpolation=cv2.INTER_AREA)
            boxes = boxes * radio
        return image,boxes, labels


# 随机截取图片的大小, 需修改
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.3, None),
            (0.5, None),
        )        

    def __call__(self, image, boxes=None, labels=None):
        mode = random.choice(self.sample_options)
        if mode is None:
            return image, boxes, labels

        height, width, _ = image.shape

        w = random.uniform(0.3 * width, 0.9 * width)
        h = random.uniform(0.05 * height, 0.2 * height)        

        sample_box_idx_lists = []        
        sample_box_idx = random.randint(len(boxes))
        sample_box_idx_lists.append(sample_box_idx)
        sample_box = boxes[sample_box_idx].copy()

        # 同时扩展CROP到其它区域, 否则只截取选择的BOX
        sample_box[0] = max(0, sample_box[0] - w/2)
        sample_box[2] = min(width, sample_box[2] + w/2)
        sample_box[1] = max(0, sample_box[1] - h/2)
        sample_box[3] = min(height, sample_box[3] + h/2)
        # print('after sample_box:', sample_box)
        # 检测剩余的BOX区域是否与挑选的BOX有交集，如有交集，则进行合并
        for idx in range(len(boxes)):
            if idx != sample_box_idx:
                if box_utils.intersects(sample_box, boxes[idx]):
                    sample_box_idx_lists.append(idx)
                    sample_box = box_utils.merge(sample_box, boxes[idx])

        x0,y0,x1,y1 = int(sample_box[0]), int(sample_box[1]), int(sample_box[2]), int(sample_box[3])
        # print(image.shape)

        crop_image = image.copy()[y0:y1,x0:x1,:]
        
        samp_boxes = boxes.copy()[sample_box_idx_lists]
        samp_labels = labels.copy()[sample_box_idx_lists]
        samp_boxes[:,(0,2)] -= x0
        samp_boxes[:,(1,3)] -= y0

        
        return crop_image, samp_boxes, samp_labels


class GTDBTransform(object):
    def __init__(self, data_root,window=1200, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.window = window
        self.data_root = data_root
        self.augment = Compose([
            ToAbsoluteCoords(window=self.window),  # ToAbsoluteCoords 转成绝对坐标，生成的box进行了缩放
            RandomSampleCrop(),
            RandomSize(),
            Mask2Windows(window=self.window),
            BackGround(data_root=self.data_root),
            ConvertFromInts(),
            ToPercentCoords(window=self.window),   # 与ToAbsoluteCoords对应，将target 恢复成x1/width, x2/width, y1/height, y2/height
            Resize(self.size)   # 将变换后图片转成 size * size
            # SubtractMeans(self.mean) 
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
