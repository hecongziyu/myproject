from __future__ import division
from ssd import build_ssd
import os
from init import init_args
import time
import logging
import datetime
from utils import helpers
from config import *
import cv2
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

IMAGE_WINDOW = 100  # 200
IMAGE_SIZE = 100   # 100

toTensor = transforms.ToTensor()

def alhpa_detect(model, image):
    # 注意对于大图片要更改大小, 后面要增加该部分处理

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    
    origin_image = image.copy()
    height, width, _ = image.shape
    # print('detect image shape:', image.shape)
    use_scale = 1.
    if width > IMAGE_WINDOW or height > IMAGE_WINDOW:
        # 注意要记住该SCALE，后面得到坐标后，还要再除掉这个SCALE，恢复到原来的大小。这里
        # 暂时没改，后面要做修改。
        use_scale = min(IMAGE_WINDOW/width,IMAGE_WINDOW/height)
        image = cv2.resize(image,(0,0),fx=use_scale, fy=use_scale, interpolation=cv2.INTER_NEAREST)



    cropped_image = np.full((IMAGE_WINDOW, IMAGE_WINDOW, image.shape[2]), 255)
    cropped_image[0:image.shape[0], 0:image.shape[1],:] = image

    # 全部转成灰度图，与训练数据相同


    img = cropped_image.astype(np.float32)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    
    img = toTensor(img)
    img = img.unsqueeze_(0)
    # print('detect convert image shape:', img.shape)
    y, boxes, scores = model(img)
    detections = y.data


    detect_type = ['alpha_location',]
    recognized_boxes = []
    scores_lists = []

    # print('recongoized boxes:', recognized_boxes, ' use scale:', use_scale, ' detections:', detections.size())
    image_lists = []
    try:
        for i, dtype in enumerate(detect_type):
            
            i += 1
            j = 0        
            while j < detections.size(2) and detections[0, i, j, 0] >= 0.01:
                 pt = (detections[0, i, j, 1:] * IMAGE_WINDOW / use_scale).cpu().numpy()
                 coords = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
                 recognized_boxes.append(coords)
                 scores_lists.append(detections[0, i, j, 0])
                 j += 1

        print('scores :',scores_lists)
        for box in recognized_boxes:
            print('detect boxes :', box)
            x0,y0,x1,y1 =  box
            if min(x0, y0, x1, y1) < 0:
                print('detect boxes apper error: ', recognized_boxes)
            else:
                aimage = origin_image[y0:y1:,x0:x1,:]
                image_lists.append(aimage)
    except Exception as e:
        print('detect error:', e)
    return image_lists


def detect_char_area(image, min_area = 80,min_y_diff=5):
    origin_image = image.copy()
    image_gray_data = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    img = image_gray_data.copy()
    blur = cv2.GaussianBlur(img, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)    
    plt.imshow(dilate)
    plt.show()
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:    
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            if (y+h)/2 > img.shape[0]*0.05 and w/h > 0.4 and cv2.contourArea(cnt) > 90:
                cnts.append([x,y,x+w,y+h, cv2.contourArea(cnt)])
    # areas = np.array(cnts,dtype=np.uint8)
    # if areas is None or len(areas) == 0:
    #     return 0,0,0,0

    image_lists = []
    for box in cnts:
        print('detect boxes :', box)
        x0,y0,x1,y1,_ =  box
        if min(x0, y0, x1, y1) < 0:
            print('detect boxes apper error: ', recognized_boxes)
        else:
            aimage = origin_image[y0:y1:,x0:x1,:]
            image_lists.append(aimage)

    
    return image_lists


def valid(args):
    # args.cfg = 'math_gtdb_512'
    args.cfg = 'ssd100'
    # weights_path = 'D:\\PROJECT_TW\\git\\data\\mathdetect\\ckpts\\weights_math_detector\\best_ssd512.pth'
    weights_path = 'D:\\PROJECT_TW\\git\\data\\ocr\\weights\\ocr_best_ssd100.pth'
    # print(args)
    cfg = exp_cfg[args.cfg]
    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        logging.debug('Using GPU with id ' + str(gpu_id))
        torch.cuda.set_device(gpu_id)

    print('cfg :', cfg)
    net = build_ssd(args, 'use', cfg, gpu_id, cfg['min_dim'], cfg['num_classes'])
    # print(net)
    mod = torch.load(weights_path,map_location=torch.device('cpu'))
    net.load_state_dict(mod)
    net.eval()    
    if args.cuda:
        net = net.cuda()    


    mean = (246,246,246)
    window = args.window
    stride = 0.01
    stepx = 200
    stepy = 400
    size = 100
    image_path = r'D:\PROJECT_TW\git\data\ocr\images\wrong\10517.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) 


    img_lists = detect_char_area(image)

    print('img lists len :', len(img_lists))
    for item in img_lists:
        item = item.astype(np.uint8)
        plt.imshow(item)
        plt.show()



if __name__ == '__main__':

    args = init_args()
    start = time.time()
    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        # logging.basicConfig(filename=filepath,
        #                     filemode='w', format='%(process)d - %(asctime)s - %(message)s',
        #                     datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
        logging.basicConfig(format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)        

        valid(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")




