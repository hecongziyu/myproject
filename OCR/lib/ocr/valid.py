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

toTensor = transforms.ToTensor()

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + u'-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text, depth=0):
        """Support batch or single str."""
        length = []
        result = []
        for str in text:
            # str = unicode(str, "utf8")    # python 3 默认为utf 8
            length.append(len(str))
            for char in str:
                # print(char)
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def normalize(image):
    # print('in -->', image.shape, ',', image, )
    image = toTensor(image)
    # print('before -->', image)
    # image.sub_(0.5).div_(0.5)
    # print('after -->', image)
    return image

def valid(net,image_path, image_height,alpha, need_detect_char=False, need_dilate=False,convert_to_bin=True,channel=1):
    converter = strLabelConverter(alpha)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(image.shape)

    if need_detect_char:
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_gray_area = image_gray.shape[0] * image_gray.shape[1]
        # print('image_gray_area -->{}'.format(image_gray_area))
        x1,y1,x2,y2 = detect_char_area(image_gray,min_area=image_gray_area*0.05,min_y_diff=5)
        if np.sum([x1,y1,x2,y2]) == 0:
            return ""
        image = image_gray[y1:y2,x1:x2]
    else:
        print('No detect_char_area')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # plt.imshow(image,'gray')
    # plt.show()
    scale =  float(image_height) / image.shape[0]
    image = cv2.resize(image,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)

    image = normalize(image)
    image = image.unsqueeze(0)

    output = net(image)
    _,preds = output.max(2)
    preds = preds.squeeze(-2)
    preds = preds.view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    words = converter.decode(preds.data, preds_size, raw=False)
    
    # print('words:', words)
    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR Evaluating Program")
    parser.add_argument('--model_name',default='ocr_best.pt', type=str, help='path of the evaluated model')
    parser.add_argument('--alpha', default='abcde')
    parser.add_argument('--image_file',default='5.png', type=str, help='path of the evaluated model')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr', type=str, help='path of the evaluated model')
    parser.add_argument("--image_height", type=int, default=32, help="图片高度")


    args = parser.parse_args()
    # torch.manual_seed(2020)
    # torch.cuda.manual_seed(2020)

    net = CRNNClassify(imgH=args.image_height,nc=1,nclass=len(args.alpha)+1,nh=256)
    net.load_state_dict(torch.load(os.path.sep.join([args.data_root,'weights',args.model_name]),map_location=torch.device('cpu')))

    # split_lists = ['A','B','C','D','E']
    # pred_result = {}
    # image_root = 'D:\\PROJECT_TW\\git\\data\\ocr\\images'
    # for split in split_lists:
    #     file_lists = os.listdir(os.path.sep.join([image_root,f'Sample{split}']))
    #     correct = 0
    #     for idx, file_name in enumerate(file_lists):
    #         try:
    #             if idx % 100 == 0:
    #                 print('{}, total {}, correct {}'.format(split, len(file_lists), correct))
    #             image_path = os.path.sep.join([image_root,f'Sample{split}', file_name])
    #             word = valid(net=net, image_path=image_path, 
    #                 image_height=args.image_height,alpha=args.alpha)
    #             # print(f'{file_name}', ' pred word ', word)
    #             if word.upper() == split:
    #                 correct += 1

    #         except Exception as x:
    #             # pass
    #             print(x)

    #     pred_result[split] = {'total':len(file_lists), 'correct':correct}
    # print('pred result:', pred_result)

    word = valid(net=net, image_path=r'D:\\img\\exam\\Sample\\bad\\15709.png', 
                 image_height=args.image_height,alpha=args.alpha)    
    print('word -->',word)