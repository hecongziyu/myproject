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
class adjustCollate(object):
    def __init__(self, imgH=32, keep_ratio=True):
        self.imgH = imgH
        self.keep_ratio = keep_ratio
        self.toTensor = transforms.ToTensor()

    def __resize__(self,image):
        image_resize = image.copy()

        if image.shape[0] != 32:
            percent = float(self.imgH) / image_resize.shape[0]
            image_resize = cv2.resize(image_resize,(0,0), fx=percent, fy=percent, interpolation = cv2.INTER_NEAREST)
        return image_resize

    def __normalize__(self,img):
        # 注意一定需要astype为np.uint8, to tensor 才会认为这是一张图片
        # img = img.astype(np.uint8)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        return img


    def __call__(self, batch):
        images, labels = zip(*batch)
        images = [self.__resize__(image) for image in images]
        PADDING_CONSTANT = 255
        assert len(set([b.shape[0] for b in images])) == 1
        # assert len(set([b.shape[2] for b in images])) == 1
        if len(images) > 0:
            dim0 = images[0].shape[0]
            dim1 = max([b.shape[1] for b in images])
            # dim2 = images[0].shape[2]
            images_padding = np.full((len(images), dim0, dim1), PADDING_CONSTANT).astype(np.uint8)

            for idx, img in enumerate(images_padding):
                # print('img shape:', img.shape, ' images shape:', images[idx].shape)
                img[:,:images[idx].shape[1]] = images[idx]
            images = images_padding
        images = [self.__normalize__(image) for image in images]
        # print(images[0])
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


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
        number = 5000
        if self.split == 'train':
            return number
        else:
            return number // 10

    def __getitem__(self, index):
        rsel = np.random.randint(6)
        if rsel in [0,1,2,3]:
            index = (index % len(self.train_dataset))
            bg_img = self.bg_image[np.random.randint(len(self.bg_image))]
            image, label = self.train_dataset[index]
        elif rsel == 4:
            index = (index % len(self.noise_dataset))
            bg_img = self.bg_image[np.random.randint(len(self.bg_image))]
            image, label = self.noise_dataset[index]
        else:
            bg_img = self.bg_image[np.random.randint(len(self.bg_image))]
            image = bg_img
            label = 'bz'

        if self.transform:

            image, label, _ = self.transform(image, label, bg_img)
        if label == 'bz':
            label = 'z'
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
    from normal_file_transform import ImgTransform
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
    random_sel = np.random.randint(0 , len(dataset), 1000).tolist()

    # for ridx, idx in  enumerate(list(range(4900, 5020))):
    for ridx, idx in  enumerate(random_sel):
        image, label = dataset[idx]
        image = image.astype(np.uint8)

        # print(idx , '--->', ' image shape:', image.shape)
        # print('target:', target.strip())
        cv2.imwrite(os.path.sep.join([args.data_root,'valid_img',f'{ridx}_{label}.png']),image)
        


