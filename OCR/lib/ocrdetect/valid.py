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

def valid(args):
    # args.cfg = 'math_gtdb_512'
    args.cfg = 'ssd300'
    # weights_path = 'D:\\PROJECT_TW\\git\\data\\mathdetect\\ckpts\\weights_math_detector\\best_ssd512.pth'
    weights_path = 'D:\\PROJECT_TW\\git\\data\\ocr\\weights\\ocr_best_ssd300.pth'
    # print(args)
    cfg = exp_cfg[args.cfg]
    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        logging.debug('Using GPU with id ' + str(gpu_id))
        torch.cuda.set_device(gpu_id)

    net = build_ssd(args, 'use', cfg, gpu_id, cfg['min_dim'], cfg['num_classes'])
    print(net)
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
    # size = 512
    size = 300
    image_path = 'D:\\PROJECT_TW\\git\\data\\ocr\\images\\wrong\\13809.png'
    # image_path = 'D:\\img\\4.png'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
    print('image shape:', image.shape)
    cropped_image = np.full((window, window, image.shape[2]), 255)
    if image.shape[0] > window:
        cropped_image[0:window, 0:window, :] = image[yl:yl+window, xl:xl+window, :]
    else:
        cropped_image[0:image.shape[0], 0:image.shape[1],:] = image

    img = cropped_image.astype(np.float32)
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    transform = transforms.ToTensor()
    img = transform(img)
    img = img.unsqueeze_(0)
    if args.cuda:
        img = img.cuda()
    print(img.size())
    start_time_n = time.time()
    y, debug_boxes, debug_scores = net(img)
    print('use time:', (time.time() - start_time_n))

    detections = y.data

    print(detections.size())
    k = 0
    i = 1
    j = 0
    recognized_boxes = []
    recognized_scores = []
    while j < detections.size(1) and detections[k, i, j, 0] >= 0.45:
        score = detections[k, i, j, 0]
        print('score :', score)
        pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
        coords = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
        recognized_boxes.append(coords)
        recognized_scores.append(score.cpu().numpy())
        j += 1

    print(recognized_boxes)

    tmp_path = 'D:\\PROJECT_TW\\git\\data\\tmp'

    for idx,box in enumerate(recognized_boxes):
        x0,y0,x1,y1 = box
        cropped_image = image[y0:y1,x0:x1,:]
        cv2.imwrite(os.path.sep.join([tmp_path,f'check_{idx}.png']),cropped_image)
        print('gen image: ', os.path.sep.join([tmp_path,f'check_{idx}.png']))
        # plt.imshow(cropped_image)
        # plt.show()

        # cv2.rectangle(image, (x0,y0,(x1-x0),(y1-y0)), (0, 0, 255), 1)


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
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)        

        valid(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")




