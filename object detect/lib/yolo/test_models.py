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
transform = transforms.ToTensor()

def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

    model = Darknet(cfg, imgsz, verbose=True)

    print('model -->', model)

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # model.fuse()
    model.to(device)
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    image = cv2.imread('./dog.jpg',cv2.IMREAD_COLOR)
    image = cv2.resize(image, (int(imgsz),int(imgsz)), interpolation=cv2.INTER_AREA)    
    source_image = image.copy()
    # image = image.astype(np.float32)
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
            cv2.rectangle(source_image, (x0,y0), (x1, y1), (0, 255, 0), 5)    

        print(f'si {si} --->', boxes, ' scores: ', score, 'classID :', classID)

    plt.imshow(source_image)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-old.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default=r'D:\PROJECT_TW\git\data\qrdetect\weights\yolov3-tiny.pt', help='weights path')
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
    print(opt)


    test(opt.cfg,
         opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.save_json,
         opt.single_cls,
         opt.augment)