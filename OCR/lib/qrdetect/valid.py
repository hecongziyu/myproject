from __future__ import division
from ssd import build_ssd
from data import *
from init import init_args
import time
import logging
import datetime
from utils import helpers
from qr_lmdb_dataset import QRDataset
from qr_lmdb_transform import QRTestTransform
from matplotlib import pyplot as plt
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')
transform = transforms.ToTensor()

def get_model(args, model_path):
    # args.cfg = 'math_gtdb_512'
    args.cfg = 'ssd300'
    # weights_path = 'D:\\PROJECT_TW\\git\\data\\mathdetect\\ckpts\\weights_math_detector\\best_ssd512.pth'
    # weights_path = 'D:\\PROJECT_TW\\git\\data\\mathdetect\\ckpts\\weights_math_detector\\best_ssd300.pth'
    weights_path = model_path
    # print(args)
    cfg = exp_cfg[args.cfg]
    gpu_id = 0
    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        logging.debug('Using GPU with id ' + str(gpu_id))
        torch.cuda.set_device(gpu_id)

    net = build_ssd(args, 'use', cfg, gpu_id, 300, 2)
    mod = torch.load(weights_path,map_location=torch.device('cpu'))
    net.load_state_dict(mod)
    net.eval()    
    if args.cuda:
        net = net.cuda()    

    return net


    # mean = (246,246,246)
    # window = 1200
    # stride = 0.01
    # stepx = 200
    # stepy = 400
    # # size = 512
    # size = 300
    # image_path = 'D:\\1044.png'
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
    # print('image shape:', image.shape)
    # cropped_image = np.full((1200, 1200, image.shape[2]), 255)
    # if image.shape[0] > window:
    #     cropped_image[0:window, 0:window, :] = image[yl:yl+window, xl:xl+window, :]
    # else:
    #     cropped_image[0:image.shape[0], 0:image.shape[1],:] = image
    # img = cropped_image.astype(np.float32)
    # img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    # transform = transforms.ToTensor()
    # img = transform(img)
    # img = img.unsqueeze_(0)
    # if args.cuda:
    #     img = img.cuda()
    # print(img.size())
    # y, debug_boxes, debug_scores = net(img)


    # detections = y.data

    # print(detections.size())
    # k = 0
    # i = 1
    # j = 0
    # recognized_boxes = []
    # recognized_scores = []
    # while j < detections.size(1) and detections[k, i, j, 0] >= 0.45:
    #     score = detections[k, i, j, 0]
    #     print('score :', score)
    #     pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
    #     coords = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
    #     recognized_boxes.append(coords)
    #     recognized_scores.append(score.cpu().numpy())
    #     j += 1

    # print(recognized_boxes)


def __get_image__(txn, image_key):
    imgbuf = txn.get(image_key.encode())
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)        
    image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_COLOR)
    return image


def __adjust_size__(img, max_width=1200):
    # print('resize img shape:', image.shape)
    if img.shape[1] > max_width or img.shape[0] > max_width:
        img = img.astype(np.uint8)
        radio = min(max_width / img.shape[1], max_width / img.shape[0])
        img = cv2.resize(img.copy(), (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
        # boxes[:,0:4] = boxes[:,0:4] * radio
    return img

def __mask_window__(img,window=1200):
    win_img = np.full((window, window, img.shape[2]), 255, dtype=np.uint8)
    win_img[0:img.shape[0],0:img.shape[1],:] = img
    return win_img


def __get_image_boxes__(model, image):
    y, debug_boxes, debug_scores = model(image)
    detections = y.data
    print(detections.size())
    k = 0
    i = 1
    j = 0
    recognized_boxes = []
    recognized_scores = []
    # while j < detections.size(1) and detections[k, i, j, 0] >= 0.05:
    while j<= detections.size(2) and detections[k, i, j, 0] >= 0.45:
        score = detections[k, i, j, 0]
        print('score :', score)
        pt = (detections[k, i, j, 1:] * args.window).cpu().numpy()
        coords = (int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))
        recognized_boxes.append(coords)
        recognized_scores.append(score.cpu().numpy())
        j += 1

    return recognized_boxes



def test_lmdb_images(args, model):

    dataset = QRDataset(data_dir=args.root_path,
                        window=1200, transform=QRTestTransform(), target_transform=None)    

    index = np.random.randint(len(dataset))

    image, boxes = dataset[index]
    image = __adjust_size__(image)
    src_image = image.copy()

    image = __mask_window__(image)

    image = cv2.resize(image.copy(), (300,300), interpolation=cv2.INTER_AREA)

    print('image shape :', image.shape)

    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
    image = image.astype(np.float32)  

    image = transform(image)
    image = image.unsqueeze(0)

    print('image size :', image.size())

    boxes = __get_image_boxes__(model, image)

    print('boxes :', boxes)

    for box in boxes:
        x0,y0,x1,y1 = box
        cv2.rectangle(src_image, (x0,y0), (x1, y1), (0, 255, 0), 2)    

    plt.imshow(src_image)
    plt.show()    


if __name__ == '__main__':

    from os.path import join

    args = init_args()
    start = time.time()
    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        # logging.basicConfig(filename=filepath,
        #                     filemode='w', format='%(process)d - %(asctime)s - %(message)s',
        #                     datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
        logging.basicConfig(format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)        

        model_path = join(args.root_path, 'ckpts', 'qr_best_ssd300.pth')
        model = get_model(args, model_path)
        model.eval()

        test_lmdb_images(args, model)


    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")




