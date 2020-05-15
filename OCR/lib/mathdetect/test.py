'''
This file contains functions to test and save the results
注意：图片大小必须是　1200 * 1200, 对于不为1200 * 1200大小的图片，在训练时
采用的是：         

cropped_image = np.full((self.window, self.window, image.shape[2]), 255)
cropped_image[:y_h-y_l, :x_h-x_l, :] = image[y_l: y_h, x_l: x_h, :]

'''
from __future__ import print_function
import os
import argparse
from init import init_args
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from utils import draw_boxes, helpers, save_boxes
import logging
import time
import datetime
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from data import *
import shutil
import torch.nn as nn
from utils.pdfutils import gen_latex_pdf,gen_latex_img_pos

"""
！！！！ 注意训练时,训练数据image的大小是1200， 再resize到500, 所以结果需要再 * 1200

实际运行时也需要将图片按1200去crop ？？？？
"""

def test_net_batch(args, net, gpu_id, dataset, transform, thresh):
    '''
    Batch testing
    '''
    num_images = len(dataset)

    if args.limit != -1:
        num_images = args.limit

    data_loader = DataLoader(dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False, collate_fn=detection_collate,
                              pin_memory=True)

    total = len(dataset)

    logging.debug('Test dataset size is {}'.format(total))

    done = 0

    for batch_idx, (images, targets, metadata) in enumerate(data_loader):

        done = done + len(images)
        logging.debug('processing {}/{}'.format(done, total))

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        y, debug_boxes, debug_scores = net(images)  # forward pass
        detections = y.data

        k = 0
        for img, meta in zip(images, metadata):

            img_id = meta[0]
            x_l = meta[1]
            y_l = meta[2]

            img = img.permute(1,2,0)
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])

            recognized_boxes = []
            recognized_scores = []

            # [1,2,200,5]
            # we only care about math class
            # hence select detections[image_id, class, detection_id, detection_score]
            # class=1 for math
            i = 1
            j = 0

            while j < detections.size(2) and detections[k, i, j, 0] >= thresh:  # TODO it was 0.6

                score = detections[k, i, j, 0]
                # 为什么要 * window , window = 1200
                pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
                coords = (pt[0] + x_l, pt[1] + y_l, pt[2] + x_l, pt[3] + y_l)
                #coords = (pt[0], pt[1], pt[2], pt[3])
                recognized_boxes.append(coords)
                recognized_scores.append(score.cpu().numpy())

                j += 1

            save_boxes(args, recognized_boxes, recognized_scores, img_id)
            k = k + 1

            if args.verbose:
                draw_boxes(args, img.cpu().numpy(), recognized_boxes, recognized_scores,
                           debug_boxes, debug_scores, scale, img_id)

def test_gtdb(args):

    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        torch.cuda.set_device(gpu_id)

    # load net
    num_classes = 2 # +1 background

    # initialize SSD
    net = build_ssd(args, 'test', exp_cfg[args.cfg], gpu_id, args.model_type, num_classes)

    logging.debug(net)
    net.to(gpu_id)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.trained_model, map_location={'cuda:1':'cuda:0'}))
    net.eval()
    logging.debug('Finished loading model!')

    dataset = GTDBDetection(args, args.test_data, split='test',
                            transform=BaseTransform(args.model_type, (246,246,246)),
                            target_transform=GTDBAnnotationTransform())

    if args.cuda:
        net = net.to(gpu_id)
        cudnn.benchmark = True

    # evaluation
    test_net_batch(args, net, gpu_id, dataset,
                   BaseTransform(args.model_type, (246,246,246)),
                   thresh=args.visual_threshold)



def test_gen_data(args):
    texts = ['让指定位置字符串中的任意位置显示不同的颜色大小','将转化为可变字符串，再根据指定字符查找该字符，再在该字符前面插入换行符']
    formuls = ['\\frac {a} {b} ', '\\sqrt {a},{b}']

    gen_latex_pdf(data_root='D:\\PROJECT_TW\\git\\data\\mathdetect', file_name='test', texts=texts, latexs=formuls)
    gen_latex_img_pos(data_root='D:\PROJECT_TW\git\data\mathdetect', file_name='test',imgH=1024)

if __name__ == '__main__':

    args = init_args()
    # start = time.time()
    # try:
    #     filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
    #     print('Logging to ' + filepath)
    #     logging.basicConfig(filename=filepath,
    #                         filemode='w', format='%(process)d - %(asctime)s - %(message)s',
    #                         datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

    #     test_gtdb(args)
    # except Exception as e:
    #     logging.error("Exception occurred", exc_info=True)

    # end = time.time()
    # logging.debug('Toal time taken ' + str(datetime.timedelta(seconds=end-start)))
    # logging.debug("Testing done!")

    test_gen_data(None)
