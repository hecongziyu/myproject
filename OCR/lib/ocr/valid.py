import argparse
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from image_utils import detect_char_area,convert_img_bin
import torchvision.transforms as transforms
from ocr_model import CRNNClassify
from torch.autograd import Variable
from utils import strLabelConverter
from matplotlib import pyplot as plt


toTensor = transforms.ToTensor()

def detect_char_pos(image_gray_data, min_area = 80,min_y_diff=5):
    img = image_gray_data.copy()
    blur = cv2.GaussianBlur(img, (7,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
    dilate = cv2.dilate(thresh, kernel, iterations=3)    
    plt.imshow(dilate)
    plt.show()
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:    
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            if (y+h)/2 > img.shape[0]*0.1 and w/h < 3:
                cnts.append([x,y,x+w,y+h, cv2.contourArea(cnt)])
    areas = np.array(cnts,dtype=np.uint8)
    print('areas len :', len(areas))
    if areas is None or len(areas) == 0:
        return 0,0,0,0
    areas_max = np.argmax(areas[:,4], axis=0)
    x1,y1,x2,y2,_ = areas[areas_max]
    return x1,y1,x2,y2



def valid(model,image, alpha, image_height=32,need_detect_char=False, need_dilate=False,convert_to_bin=True,channel=1):
    converter = strLabelConverter(alpha)

    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    x0,y0,x1,y1 = detect_char_pos(image_gray)
    image = image[y0:y1,x0:x1,]

    scale =  float(image_height) / image.shape[0]

    image = cv2.resize(image,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(image)
    plt.show()

    image.astype(np.float32)
    image = toTensor(image)
    image = image.unsqueeze(0)

    print('image size :', image.size())

    with torch.no_grad():
        preds = model(image).cpu()
        _, preds = preds.max(2)
        # print('preds size :', preds.size())
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        words = converter.decode(preds.data, preds_size.data, raw=False)
        print('words :', words)

    # image = normalize(image)
    # image = image.unsqueeze(0)

    # output = net(image)
    # _,preds = output.max(2)
    # preds = preds.squeeze(-2)
    # preds = preds.view(-1)
    # preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # words = converter.decode(preds.data, preds_size, raw=False)
    
    # # print('words:', words)
    # return words


if __name__ == '__main__':
    from os.path import join
    from normal_lmdb_dataset import lmdbDataset
    from normal_lmdb_transform import ImgTransform
    

    parser = argparse.ArgumentParser(description="OCR Evaluating Program")
    parser.add_argument('--model_name',default='ocr_number_best.pt', type=str, help='path of the evaluated model')
    parser.add_argument('--alpha', default='0123456789z')
    parser.add_argument('--file_name', default='1_1.png', type=str)
    parser.add_argument('--data_root',default=r'D:\PROJECT_TW\git\data\ocr\number', type=str, help='path of the evaluated model')
    parser.add_argument("--image_height", type=int, default=32, help="图片高度")


    args = parser.parse_args()
    # torch.manual_seed(2020)
    # torch.cuda.manual_seed(2020)

    model = CRNNClassify(imgH=args.image_height,nc=1,nclass=len(args.alpha)+1,nh=256)
    model.load_state_dict(torch.load(join(args.data_root,'ckpts',args.model_name),map_location=torch.device('cpu')))
    model.eval()
    print('model :', model)

    dataset = lmdbDataset(root=args.data_root, split='valid',transform=ImgTransform(data_root=args.data_root))

    index = 43

    image, target = dataset[index]

    image = cv2.imread(join(args.data_root,'test_img', args.file_name), cv2.IMREAD_COLOR)

    print('image :', image.shape)

    valid(model=model, image=image, alpha=args.alpha)
