import argparse
import json
from torch.utils.data import DataLoader
import cv2
from models import *
from utils.datasets import *
from utils.utils import *
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.autograd import Variable
import time
from data.qrtransform import QRTransform
from torch.autograd import Variable
from data.qrdataset import QRDataset,collate_fn,AnnotationTransform
transform = transforms.ToTensor()
from matplotlib import pyplot as plt
import cv2 
import numpy as np


imgsz = (416,416)

def get_model(opt):
    model = Darknet(opt.cfg, imgsz, verbose=True)
    device = 'cpu'
    ckpt = torch.load(opt.ckpts, map_location='cpu')
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device)
    model.eval()
    return model

def valid_image_dataset(model, opt):
    dataset = QRDataset(data_dir=opt.data_root,
                        window=416, transform=QRTransform(window=416), target_transform=AnnotationTransform())    
    conf_thres=0.5
    iou_thres=0.5  # for nms

    image, labels = dataset[0]
    image = image.astype(np.uint8)
    print('source image size :', image.shape)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,  # Shuffle=True unless rectangular training is used
                                             pin_memory=False,
                                             collate_fn=collate_fn)  

    for i, (imgs, targets) in enumerate(dataloader):                                              
        print('idx %s imgs size %s' % (i, imgs.size()))
        imgs = imgs.float() / 255.0    
        with torch.no_grad():
            print('imgs :', imgs)
            inf_out, train_out = model(imgs, verbose=True)

            # inf_out[torch.where(torch.isnan(inf_out))] = 0

            print('infert output size :', inf_out.size())
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=False)

            print('infer output :', len(output))
            # boxes = output[0][:, :4].clone().tolist()
            # print('score :', boxes[:,5])
            for si, pred in enumerate(output):
                print('pred :', pred)
                clip_coords(pred, (416, 416))

                boxes = pred[:, :4].clone().tolist()
                score = pred[:,4].clone().tolist()
                classID = pred[:,5].clone().tolist()

                for bitem in boxes:
                    x0,y0,x1,y1 = bitem
                    x0 = int(x0)
                    y0 = int(y0)
                    x1 = int(x1)
                    y1 = int(y1)
                    cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 1)    
            print(f'si {si} --->', boxes, ' scores: ', score, 'classID :', classID)

            break

    plt.imshow(image)
    plt.show()


def valid_image_file(model, opt, file_name=None, imgsz=416):
    conf_thres=0.01
    iou_thres=0.5  # for nms
    multi_label=False
    device = 'cpu'
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = cv2.imread(r'D:\PROJECT_TW\git\data\qrdetect\images\real\source\00d78ebc-24ff-42b0-ab46-2250401530b7.jpg',cv2.IMREAD_COLOR)
    image = mask_image(image)
    image = image.astype(np.uint8)

    # image = cv2.resize(image, (int(imgsz),int(imgsz)), interpolation=cv2.INTER_AREA)


    source_image = image.copy()

    # image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image / 255.0    
    image = transform(image)
    image = image.unsqueeze(0)    
    image = image.to(device)
    image = Variable(image.type(Tensor), requires_grad=False)
    print('image size :', image.size(), ':', image.dtype)


    with torch.no_grad():
        # inf_out, train_out = model(torch.zeros((1, 3, imgsz, imgsz), device=device), verbose=True) 
        start_time = time.time()
        
        inf_out, train_out = model(image, verbose=True)
        print('use time :', (time.time() - start_time))
        print('inf out size :', inf_out.size())
        print('train out size :', train_out[0].size(), train_out[1].size())

    print('out class :', inf_out.size())
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
    # output = np.array(output[0])
    # print('out put:', output.shape)

    for si, pred in enumerate(output):
        clip_coords(pred, (imgsz, imgsz))
        boxes = pred[:, :4].clone().tolist()
        score = pred[:,4].clone().tolist()
        classID = pred[:,5].clone().tolist()

        for bitem in boxes:
            x0,y0,x1,y1 = bitem
            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x1)
            y1 = int(y1)
            cv2.rectangle(source_image, (x0,y0), (x1, y1), (0, 255, 0), 1)    

        print(f'si {si} --->', boxes, ' scores: ', score, 'classID :', classID)

    plt.imshow(source_image)
    plt.show()


def mask_image(image, imgsz=416):
    radio = min(imgsz / image.shape[1], imgsz / image.shape[0])
    image = cv2.resize(image.copy(), (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
    win_img = np.full((imgsz, imgsz, 3), 255)
    win_img[0:image.shape[0],0:image.shape[1],:] = image.copy()
    return win_img

def nms_suppression(prediction, conf_thres=0.1, iou_thres=0.6, multi_label=True, classes=None, agnostic=False):
    min_wh, max_wh = 2, 60  # (pixels) minimum and maximum box width and height
    nc = prediction[0].shape[1] - 5  # number of classes




# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    print(dir(net))
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]    

def valid_opencv(opt):
    conf_thres=0.1
    iou_thres=0.5  # for nms
    multi_label=False

    net = cv2.dnn.readNetFromDarknet(opt.cfg, opt.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    image = cv2.imread(r'D:\PROJECT_TW\git\data\qrdetect\images\real\source\00d78ebc-24ff-42b0-ab46-2250401530b7.jpg',cv2.IMREAD_COLOR)
    image = mask_image(image)
    src_img = image.copy()
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    

    # plt.imshow(image)
    # plt.show()

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), [0, 0, 0], swapRB=False, crop=False)
    net.setInput(blob)
    print('getOutputsNames(net):', getOutputsNames(net))
    outputs = net.forward(getOutputsNames(net))

    print('outputs shape :', len(outputs))

    # print('type inf output :', type(inf_out), inf_out.shape)
    # print('type train out :', type(train_out), train_out.shape)
    confidences = []
    boxes = []

    for layers in outputs:
        print('layers shape:', layers.shape)

        for detect in layers:
            score = detect[4]
            # print('score :', score)
            if score > conf_thres:
                print('detect :' , detect, ' scores: ', score)
                center_x =  int(detect[0] * 416)
                center_y =  int(detect[1] * 416)
                width = int(detect[2] * 416)
                height = int(detect[3] * 416)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(score))
                boxes.append([left, top, width, height])

                # x0 = max(0, int((x - w /2 ) * 416))
                # y0 = max(0, int((y - h /2) * 416))
                # x1 = min(416, int((x + w/2) * 416))
                # y1 = min(416, int((y + h /2) * 416))
                # print('x0:%s, y0:%s, x1:%s, y1:%s' % (x0,y0,x1,y1))
                # cv2.rectangle(src_img, (x0,y0), (x1, y1), (0, 255, 0), 2)                    
                # x0,y0,x1,y1 = xywh2xyxy(detect[0:4])

    print('len boxes :', len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)
    print('len indices:', indices)

    for idx in indices:

        x0,y0,w,h = boxes[idx[0]]
        x1 = x0 + w
        y1 = y0 + h
        cv2.rectangle(src_img, (x0,y0), (x1, y1), (0, 255, 0), 2)                    
    plt.imshow(src_img)
    plt.show()
    # output = nms_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=False)

    # boxes_lists = []
    # conf_lists = []

    # for si, pred in enumerate(output):
    #     boxes = pred[:, :4].clone().tolist()
    #     score = pred[:,4].clone().tolist()
    #     conf_lists.extend(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolo-fastest.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\qrdetect', type=str, help='path of the math formula data')
    parser.add_argument('--ckpts', type=str, default=r'D:\PROJECT_TW\git\data\qrdetect\ckpts\yolo3_tiny_best.pth', help='*.data path')
    parser.add_argument('--weights', type=str, default=r'D:\PROJECT_TW\git\data\qrdetect\ckpts\yolo3_tiny_best.weights', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    # opt.data = check_file(opt.data)  # check file
    # print(opt)

    # model = get_model(opt)

    # valid_image_dataset(model, opt)
# 
    # valid_image_file(model,opt)

    valid_opencv(opt)


