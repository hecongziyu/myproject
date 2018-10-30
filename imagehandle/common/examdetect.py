# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
from .config import cfg
from .exceptions import ExampleException
import logging
from math import fabs,sin,radians,cos


DEBUG = cfg.BASE.DEBUG
logger = logging.getLogger('example')


class EXDetect(object):
    def __init__(self, id, image, binaray=True, mobile_type='default'):
        self.id = id
        # 原图
        self.source_imsage = None
        # 经过处理完成的灰度图（包括调整大小、角度等处理）
        self.base_config = cfg
        self.ext_config = cfg.IMAGE
        self.blue_val = 0
        self.angle = 0
        self.center_y = 0       # 试卷中心区域Y轴  用于后期切割大题用。
        self.center_heigh = 0   # 试卷中心区域高
        self.center_x = 0       # 试卷中心区域X轴  用于后期切割大题用。
        self.center_width = 0   # 试卷中心区域宽
        self.answer_lines = None  # 填空题横线队列 [Y,X,H,W], 在clip_answers时设值
        self.origin_image,self.center_img, self.gray_image, self.center_gray_img = self._init_image(image, binaray)
        self.image_height, self.image_width = self.gray_image.shape



    def _init_image(self, image, binaray=True):
        if binaray:
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            if DEBUG:
                plt.imshow(image,'brg')
                plt.show()
            self.source_imsage = image
            # 检测图片大小是否符合规格，检测图片清晰度
            blue_val, high, width, angle, self.center_x,self.center_width,self.center_y,self.center_heigh = self.__check_image__(image)
            logger.info("试卷ID {} 原始图片 清晰度 {:.3f} 高度 {} 宽度 {} 中心宽度 {} 倾斜角度 {:.3f}".format(self.id, blue_val,
                                                                                         high, width, self.center_width, angle))
            self.blue_val = blue_val
            self.angle = angle

            if blue_val < cfg.IMAGE.BLUR_THRESHOLD:
                logger.info('试卷ID {} 模糊度 {} 小于 {}.'.format(self.id, blue_val, cfg.IMAGE.BLUR_THRESHOLD))
                raise ExampleException(cfg.RET.ERROR_BLUR[0], cfg.RET.ERROR_BLUR[1])

            if self.center_width < 1600:
                logger.info('试卷ID {} 内容宽度太小 试卷宽度为 {}.'.format(self.id, self.center_width))
                raise ExampleException(cfg.RET.ERROR_HIGH[0], cfg.RET.ERROR_HIGH[1])
            
            if high < cfg.IMAGE.MIN_HIGH:
                logger.info('试卷ID {} 高度 {} 小于 {}.'.format(self.id, high, cfg.IMAGE.MIN_HIGH))
                raise ExampleException(cfg.RET.ERROR_HIGH[0], cfg.RET.ERROR_HIGH[1])


            if abs(angle) > cfg.IMAGE.MIN_LEAN_ANGLE:
                logger.info('试卷ID {} 倾斜角度 {:.3f} 大于 {}.'.format(self.id, angle, cfg.IMAGE.MIN_LEAN_ANGLE))
                raise ExampleException(cfg.RET.ERROR_LEAN_ANGLE[0], cfg.RET.ERROR_LEAN_ANGLE[1])

            if abs(angle) > cfg.IMAGE.CORRECT_LEAN_ANGLE:
                logger.info("试卷倾斜角度大于 {} 纠正图片倾斜角度, 倾斜角度为{}.".format(cfg.IMAGE.CORRECT_LEAN_ANGLE, angle))
                image = self.adjust_image_angle(image, angle)

            # if high > cfg.IMAGE.MIN_HIGH:
            #     logger.info("试卷高度大于 {} 重新设置图片大小.".format(cfg.IMAGE.MIN_HIGH))
            #     image = self.adjust_image_size(image)

            # 根据中心宽度对试卷进行调整
            if float(self.center_width/1600) >= 1.1:
                a_width = width - (self.center_width - 1600)
                if a_width > 1860:
                    a_heigh = int(a_width  * high / width)
                    # 中心位置做调整， 根据重设大小，按比例调整试卷中心位置
                    self.center_y = int(self.center_y * a_width/width) - 5
                    self.center_heigh = int(self.center_heigh * a_width/width) + 10
                    self.center_x = int(self.center_x * a_width/width ) - 5
                    self.center_width = int(self.center_width * a_width/width) + 10
                    # degree = high/width
                    logger.info("试卷宽度 {} 试卷中心宽度 {} 大于1600 重新设置图片大小, 重新设置后的宽度为 {}.".format(width,self.center_width, a_width))
                    image = self.adjust_image_size(image, a_width,a_heigh)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        center_img = image[self.center_y:self.center_y+self.center_heigh, self.center_x:self.center_x+self.center_width]
        center_gray_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
        return image, center_img, gray, center_gray_img


    # 检测图片的高、宽、模糊度等等, 对有问题的抛出异常
    def __check_image__(self,image):
        # 模糊度检测
        blue_val = cv2.Laplacian(image, cv2.CV_64F).var()
        high, width = image.shape[:2]
        img_x, img_y, img_w, img_h = self.__get_img_center__(image)
        logger.debug('试卷中心区域位置信息 X {}  Y {} 宽 {} 高 {}'.format(img_x,img_y,img_w,img_h))
        angle = self.get_image_angle(image[img_y:img_y+img_h, img_x:img_x+img_w])
        return blue_val, high, width, angle,img_x, img_w, img_y, img_h

    # 取得试卷图片中心区域的高、宽，用于调整图片大小
    def __get_img_center__(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        # ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # img = cv2.blur(img,tuple((5,5)))
        nimg, contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        rcnt = []
        rarea = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5000:
                rarea.append(cv2.contourArea(cnt))
                rcnt.append(cnt)
        pos = np.argmax(rarea)
        cnt = rcnt[pos]
        img_x, img_y, img_w, img_h = cv2.boundingRect(cnt)
        if DEBUG:
            cv2.rectangle(image.copy(), (img_x, img_y), (img_x + img_w, img_y + img_h), 0, -1)
            plt.imshow(image, 'brg')
            plt.show()

        return img_x, img_y, img_w, img_h

    # 调整图片大小
    def adjust_image_size(self, image, width, high):
        newimg = cv2.resize(image, (width, high), cv2.INTER_CUBIC)
        return newimg


    # 得到图片的倾斜角度
    def get_image_angle(self, image):
        angle = 0
        newimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if DEBUG:
            plt.imshow(newimg,'gray')
            plt.show()
        # newimg = cv2.bitwise_not(newimg)
        newimg = cv2.GaussianBlur(newimg, (5, 5), 0)

        newimg = cv2.adaptiveThreshold(newimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY_INV, 11, 2)  # 图片转成二进制图片

        if DEBUG:
            plt.imshow(newimg,'gray')
            plt.show()


        lines = cv2.HoughLinesP(newimg, 1, np.pi / 180, 80,
                                minLineLength=self.ext_config.MIN_LINE_LENGTH, maxLineGap=self.ext_config.MAX_LINE_GAP)

        lines = lines[:, 0, :]
        number = 0
        if len(lines) < 4:
            raise ExampleException(99, '错误的试卷格式，没有找到试卷边框')
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            k = (y1 - y2) / (x1 - x2 + 0.000001)
            width = abs(x2 - x1)
            if width > self.ext_config.MIN_LINE_WIDTH:
                # math.atan(k)*180/3.1415926 为角度，
                number = number + 1
                angles.append(math.atan(k)*180/3.1415926)
                # print(x1,':',x2,':',width,':',y1,':',y2,':', math.atan(k)*180/3.1415926)
                # summary = summary + math.atan(k)*180/3.1415926
        # if number > 0:
        #     angle = summary / number
        
        if len(angles) > 4:
            angle = np.mean(np.array(angles))
        return angle

    # 调整图片的角度，恢复到正常角度
    def adjust_image_angle(self, image, angle):
        # high, width = image.shape[:2]
        # center = ((width // 2, high // 2))
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # newimg = cv2.warpAffine(image, M, (width, high), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        height, width = image.shape[:2]
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
        matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步
        imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        return imgRotation



    # 试卷题目边框将试卷分割
    def clip_exam(self):
        if DEBUG:
            logger.debug('中心区域图片')
            plt.imshow(self.center_gray_img,'gray')
            plt.show()

        image = np.copy(self.center_gray_img)
        img = cv2.GaussianBlur(image, (5, 5), 0)
        # ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if DEBUG:
            logger.debug('调试，显示图片二进制图')
            plt.imshow(bw,'gray')
            plt.show()

        horizontal = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = int(cols / 30)

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        regions = self._get_rect(horizontal)
        return regions

    # 提取边框的位置
    def _get_rect(self,image):

        # 线加组
        image = cv2.blur(image, tuple((5, 5)))
        element = cv2.getStructuringElement(cv2.MORPH_RECT, tuple((21, 21)), (-1, -1))
        imgr = cv2.dilate(image, element, iterations=1)
        if DEBUG:
            #             logger.debug('调试，显示处理后合并图')
            plt.imshow(imgr, 'gray')
            plt.show()

        nimg, contours, hierarchy = cv2.findContours(imgr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cnts.append([y, x,  h ,w])
        cnts = np.array(cnts)
        idex = np.lexsort([cnts[:, 1], cnts[:, 0]])
        cnts = cnts[idex,:].tolist()


        self.answer_lines = []
        for cnt in cnts:
            if (cnt[3] > int(self.center_width*0.2)) and (cnt[3] < int(self.center_width * 0.9)):
                self.answer_lines.append([cnt[0]+self.center_y, cnt[1] + self.center_x, cnt[2], cnt[3]])
        print(self.answer_lines)

        cnts = [x for x in cnts if x[3] >= int(self.center_width*0.9)]

        regions = []
        for i in range(len(cnts) - 1):
            regions.append([cnts[i][0] + self.center_y, cnts[i][1] + self.center_x, cnts[i+1][0] - cnts[i][0], cnts[i][3]])

        print(regions)
        return  regions



