# encoding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt


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

        return image.astype(np.float32), bg_image, label



# 将多图合并到前景图片上面
class MergeImage(object):
    def __init__(self, min_radio=0.3, max_radio=0.8):
        self.min_radio = min_radio
        self.max_radio = max_radio

    def __adjust_image_size__(self, image, bg_image, label):
        prob = (image.shape[0] * image.shape[1]) / (bg_image.shape[0] * bg_image.shape[1])
        if len(label) == 1:
            pred_prob = np.random.uniform(self.min_radio, self.max_radio)
        else:
            pred_prob = np.random.uniform(0.4, 0.8)

        scale = min(np.sqrt(pred_prob/prob), bg_image.shape[0]/image.shape[0], bg_image.shape[1]/image.shape[1])
        # print('scale:', scale)
        # scale = max(0.6, scale)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image

    def __custom_erosion__(self, image, median=214):

        
        # in_image = image.copy()

        # # _INV
        # ret, binary =  cv2.threshold(in_image, 0, 255, cv2.THRESH_BINARY)     
        # in_image = binary.copy()

        # # dilate MORPH_RECT  MORPH_CROSS MORPH_ELLIPSE
        # dilatation_type = cv2.MORPH_CROSS
        # dilatation_size = 1
        # element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        # dilatation = cv2.dilate(in_image, element)
        # in_image = dilatation.copy()

        # # erode
        # # cv.MORPH_RECT  cv.MORPH_CROSS cv.MORPH_ELLIPSE
        # erosion_type = cv2.MORPH_ELLIPSE
        # erosion_size = 1        
        # element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))        
        # erosion = cv2.erode(in_image,element,iterations = 2)
        # in_image = erosion.copy()


        
        # in_image = 255 -in_image
        # # in_image[np.where(in_image==255)] = 128
        # return in_image

        return image

    def __call__(self, char_img, bg_image, label):

        if bg_image is None:
            return char_img, bg_image, label

        image = self.__adjust_image_size__(char_img.copy(), bg_image, label)

        mask = np.ones(image.shape, image.dtype) * 255
        height, width, _ = image.shape
        b_height,b_width,_ = bg_image.shape


        center_y = np.random.randint(int(height/2) , int(b_height-(height/2))) if b_height > height else int(height/2)
        center_x = np.random.randint(int(width/2) , int(b_width-(width/2))) if b_width > width else int(width/2)

        center = (center_x,center_y)

        # MONOCHROME_TRANSFER
        image = cv2.seamlessClone(image,bg_image, mask,center, cv2.MONOCHROME_TRANSFER)

        # mask = np.ones(image.shape, image.dtype) * 255
        # image = cv2.textureFlattening(image, mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
        return image, bg_image, label

class ClipMergeImage(MergeImage):
    # https://www.jianshu.com/p/49adfbe4b804

    def __init__(self,min_radio=0.1, max_radio=0.25):
        super(ClipMergeImage,self).__init__(min_radio=min_radio, max_radio=max_radio)

    def __call__(self, char_img, bg_image, label):
        if bg_image is None:
            return char_img, bg_image, label

        image = char_img.copy()
        # print('bg image shape:', bg_image.shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)

        # 当前图片面积在背景图片中的占比
        # image = self.__adjust_image_size__(image, bg_image)
        image = self.__adjust_image_size__(image, bg_image,label)

        height,width = bg_image.shape
        y, x = np.where(image != 0)
        offset_y = np.random.randint((height - np.max(y))) #int((height/2) - np.mean(y)) -1
        offset_x = np.random.randint((width - np.max(x))) #int((width/2) - np.mean(x)) -1
        y = y + offset_y
        x = x + offset_x
        image = bg_image.copy()
        image[y,x] = np.random.randint(20,120)
        return image,  bg_image, label

class CombinCharImages(object):
    def __init__(self, mean=None):
        self.mean = mean
    def __call__(self, image, bg_image, label):
        # if np.random.randint(2):
        #     image = [image,image]
        if type(image) == list:
            height = np.max([x.shape[0] for x in image])
            width = np.sum([x.shape[1] for x in image])
            cm_image = np.zeros((height,width, image[0].shape[2]))
            cm_image = cm_image.astype(image[0].dtype)

            if self.mean is None:
                cm_image[:,:,0] = np.median(bg_image[:,:,0])
                cm_image[:,:,1] = np.median(bg_image[:,:,1])
                cm_image[:,:,2] = np.median(bg_image[:,:,2])
            else:
                # print('mean:', self.mean)
                cm_image = cm_image * self.mean
            
            offset_x = 0
            offset_y = 0

            for idx,img in enumerate(image):
                height, width , _ = img.shape
                # print('--->', img.shape, ':', offset_y,':', offset_y+height)
                if idx == 0:
                    offset_x_bias = 0
                else:
                    offset_x_bias = np.random.randint(6)
                    # offset_x_bias = 6
                cm_image[offset_y:offset_y+height, offset_x-offset_x_bias:offset_x+width-offset_x_bias,:] = img
                offset_x = width + offset_x - offset_x_bias
                # offset_y = height
            image = cm_image.copy()
            # print('image shape:', image.shape)
        return image, bg_image, label

        

class ExpandBg(object):
    def __init__(self, min_width=60, min_height=32):
        self.min_width = min_width
        self.min_height = min_height

    def __call__(self, image, bg_image, label):
        if bg_image is not None:
            height, width, channel = bg_image.shape
            if bg_image.shape[0] < self.min_height or bg_image.shape[1] < self.min_width:
                # print('expand image ', bg_image.shape, ' mean :', np.mean(bg_image[:,:,0]),":",np.mean(bg_image[:,:,1]))
                exp_img = np.zeros((max(self.min_height,bg_image.shape[0]) + 1, 
                                    max(self.min_width, bg_image.shape[1]) + 1, bg_image.shape[2])) 
                exp_img[:,:,0] = np.median(bg_image[:,:,0])
                exp_img[:,:,1] = np.median(bg_image[:,:,1])
                exp_img[:,:,2] = np.median(bg_image[:,:,2])

                exp_img = exp_img.astype(bg_image.dtype)
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

class CharImgTransform(object):
    '''
    将原图与背景图进行混合的处理
    '''
    def __init__(self, imgH=32):
        self.imgH = imgH
        self.augment = Compose([
            CombinCharImages(),     # 合并有多个字符串的截图
            ExpandBg(),             # 背景图扩展，并加入噪声
            MergeImage(),           # 混合图片，将字符串截图与背景图进行合并
            ConvertFromInts(),      # 检测图像是否为BGR，如是则转成灰度模式
            # RandomContrast(),
            RandomBrightness()
        ])

    def __call__(self, char_img, bg_image, label):
        return self.augment(char_img, bg_image, label)


class ClipCharImgTransform(object):
    '''
    将切割的图片与背景图进行混合的处理
    '''
    def __init__(self, imgH=32):
        self.imgH = imgH
        self.augment = Compose([
            CombinCharImages(mean=255),
            ExpandBg(),
            ClipMergeImage(),
            ConvertFromInts(),
            # RandomContrast(),
            RandomBrightness()
        ])

    def __call__(self, char_img, bg_image, label):
        return self.augment(char_img, bg_image, label)





