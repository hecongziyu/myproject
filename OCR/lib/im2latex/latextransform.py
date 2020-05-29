import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import os
import pickle as pkl
from build_vocab import PAD_TOKEN, UNK_TOKEN

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
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, latex=None):
        return self.lambd(img, latex)


class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32)


# class ToAbsoluteCoords(object):
#     def __call__(self, image, latex=None):
#         height, width, channels = image.shape
#         boxes[:, 0] *= width
#         boxes[:, 2] *= width
#         boxes[:, 1] *= height
#         boxes[:, 3] *= height

#         return image, boxes, labels


# class ToPercentCoords(object):
#     def __call__(self, image, latex=None):
#         height, width, channels = image.shape
#         boxes[:, 0] /= width
#         boxes[:, 2] /= width
#         boxes[:, 1] /= height
#         boxes[:, 3] /= height

#         return image, boxes, labels


class GenResize(object):
    def __init__(self, imgH=64):
        self.imgH = imgH

    def __call__(self, image):
        height, width, _ = image.shape
        if height > self.imgH:
            radio = self.imgH/height
            image = cv2.resize(image, (int(width*radio),self.imgH), interpolation=cv2.INTER_AREA)
        return image


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 255.0] -= 255.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 255.0
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, latex=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class ToTensor(object):
    def __call__(self, cvimage, latex=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):

        if random.randint(2):
            return image

        im = image.copy()
        # im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return self.rand_light_noise(im)


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image):
        if random.randint(2):
            return image

        height, width, depth = image.shape
        ratio = random.uniform(1, 1.2)
        left = random.uniform(0, int(width*ratio) - width)
        top = random.uniform(0, int(height*ratio) - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image
        return image

'''替换前景景图片'''
class BackGround(object):
    def __init__(self, data_root, back_data_dir='bg'):
        self.data_root = data_root
        self.back_data_dir = back_data_dir
    def __call__(self, image):
        if random.randint(3) == 0:
            return image        
        height,width, _ = image.shape
        bg_files = os.listdir(os.path.sep.join([self.data_root,self.back_data_dir]))
        bg_img_file = os.path.sep.join([self.data_root, self.back_data_dir, bg_files[random.randint(0, len(bg_files))]])
        bg_img = cv2.imread(bg_img_file,cv2.IMREAD_UNCHANGED)
        bg_img = cv2.resize(bg_img, (width,height), interpolation=cv2.INTER_AREA)
        # bg_img[np.where(image<=246)] = np.random.randint(0,16,len(np.where(image<=246)[0]))
        bg_img[np.where(image<=246)] = 0
        return bg_img

'''文字膨胀处理, 暂不处理'''
class TxtDilate(object):
    pass 


'''文字模糊处理，暂不处理'''
class TxtBlur(object):
    pass      


class RandomSize(object):
    def __init__(self, min_radio=0.8, max_radio=1):
        self.min_radio = min_radio
        self.max_radio = max_radio


    def __call__(self, image):
        if random.randint(2):
            radio = random.uniform(self.min_radio, self.max_radio)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width*radio),int(height * radio)), interpolation=cv2.INTER_AREA)
        return image

class ExpandWidth(object):
    def __init__(self, imgH, max_width):
        self.imgH = imgH
        self.max_width = max_width

    def __call__(self, image):

        image_ext = np.ones((self.imgH, self.max_width, image.shape[2]),dtype=image.dtype) * PAD_TOKEN
        image_ext[0:image.shape[0], 0:image.shape[1],:] = image
        return image_ext



class LatexImgTransform(object):
    def __init__(self, imgH=64, max_width=600,mean=(104, 117, 123), data_root='./data'):
        self.mean = mean
        self.imgH = imgH
        self.data_root = data_root
        self.max_width = max_width
        # print('mean :', self.mean)

        self.augment = Compose([
            ConvertFromInts()
            # ExpandWidth(self.imgH, self.max_width)
            # RandomSize(),
            # PhotometricDistort(), # PhotometricDistort 给数据增加噪声
            # Expand(self.mean),     # 随机扩展图片, mean 中间颜色
            # RandomSampleCrop(),
            # GenResize(self.imgH)   # 将变换后图片转成 size * size, 注意gen resize需要修改，因为图片放大后，宽度会超出600，这种情况会有问题

            # BackGround(self.data_root)
        ])

    def __call__(self, img):
        return self.augment(img)
