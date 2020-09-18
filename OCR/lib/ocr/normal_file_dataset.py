# encoding: utf-8
import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import six
import torch
import os
from os.path import join


class OcrFileDataset(Dataset):
    def __init__(self, root=None,split='train', transform=None, max_len=150):
        self.transform = transform
        self.split = split
        self.max_len = max_len
        self.data_root = root

        # 背景图片
        self.bg_image = self.__load_bg_image__()
        # 训练数据
        self.train_dataset = self.__load_train_image__()
        # 噪声数据
        self.noise_dataset = self.__load_noise_image__()


    def __load_bg_image__(self):
        file_lists = os.listdir(join(self.data_root,'bg'))
        img_lists = []
        for fitem in file_lists:
            image = cv2.imread(join(self.data_root,'bg', fitem), cv2.IMREAD_COLOR)
            img_lists.append(image)
        return img_lists


    def __load_train_image__(self):
        file_lists = os.listdir(join(self.data_root,'fimages'))
        img_lists = []
        for fitem in file_lists:
            image = cv2.imread(join(self.data_root,'fimages', fitem), cv2.IMREAD_COLOR)
            label = fitem.split('_')[0]
            img_lists.append((image,label))
        return img_lists

    def __load_noise_image__(self):
        file_lists = os.listdir(join(self.data_root,'eimages'))
        img_lists = []
        for fitem in file_lists:
            image = cv2.imread(join(self.data_root,'eimages', fitem), cv2.IMREAD_COLOR)
            label = fitem.split('_')[0]
            img_lists.append((image,'z'))
        return img_lists


    def __len__(self):
        return 100

    def __getitem__(self, index):
        index = (index % len(self.train_dataset))
        image, label = self.train_dataset(index)
        return image, label
        # # assert index <= len(self), 'index range error'
        # if index < self.nSamples:
        #     with self.env.begin(write=False) as txn:
        #         image, target = self.pull_train_item(txn, index)
        #         print('real img :', image.shape)
        # else:
        #     image = None
        #     target = 'z'
        # target = "".join(target.split()[0:self.max_len])

        # if self.transform:
        #     image, target = self.transform(image, target)

        # return image, target

if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    from functools import partial
    import time
    from torch.utils.data.sampler import SubsetRandomSampler
    from normal_lmdb_transform import ImgTransform
    from torch.utils.data import DataLoader
    import torchvision

    parser = argparse.ArgumentParser(description='latex imdb dataset')
    parser.add_argument('--data_root',default=r'D:\\PROJECT_TW\\git\\data\\ocr\\special', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='train', type=str)    
    parser.add_argument('--batch_size', default=16, type=int)    
    parser.add_argument("--cuda", action='store_true',default=True, help="Use cuda or not")   
    parser.add_argument("--shuffle", action='store_true',default=True, help="Use cuda or not")  
    args = parser.parse_args()

    def imshow(img,text=None,should_save=False): 
        #展示一幅tensor图像，输入是(C,H,W)
        npimg = img.numpy() #将tensor转为ndarray
        npimg = npimg.astype(np.uint8)
        print('image shape:', npimg.shape)
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
        plt.show()        


    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    dataset = OcrFileDataset(root=args.data_root, split='train',transform=ImgTransform(data_root=args.data_root))

    dataset_size = len(dataset)

    print('dataset size:', dataset_size)
    random_sel = np.random.randint(0 , len(dataset), 5).tolist()

    # for ridx, idx in  enumerate(list(range(4900, 5020))):
    for ridx, idx in  enumerate(random_sel):
        image, label = dataset[idx]
        image = image.astype(np.uint8)

        # print(idx , '--->', ' image shape:', image.shape)
        # print('target:', target.strip())
        cv2.imwrite(os.path.sep.join([args.data_root,'valid_img',f'{ridx}_{target}.png']),image)
        


