# encoding: utf-8
import random
import sys
import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import six
import torch
from image_utils import detect_char_area
from torch.utils.data import DataLoader

BLANK_FLAG = u'z'

class resizeNormalize(object):
    def __init__(self, size):
        self.size = size
        self.toTensor = transforms.ToTensor()
    def __call__(self, image):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
        image = self.toTensor(image)
        # image.sub_(0.5).div_(0.5)
        return image

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



class alignCollate(object):
    def __init__(self, imgH=32, imgW=128, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        # import ipdb
        # ipdb.set_trace()
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.shape
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        print(images.size())
        return images, labels



class lmdbDataset(Dataset):
    def __init__(self, root=None,split='train', transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.transform = transform

        self.target_transform = target_transform
        self.split = split
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            if self.split == 'train':
                nSamples = int(txn.get('train_num_samples'.encode()))
            else:
                nSamples = int(txn.get('valid_num_samples'.encode()))
            self.nSamples = nSamples



    def __len__(self):
        return int(self.nSamples/10)
        # return 100
        # return 20

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            if self.split == 'train':
                image, label = self.pull_train_item(txn, index)
            else:
                image, label = self.pull_valid_item(txn, index)
        
        return image, label



    def __get_image__(self, txn, image_key):
        imgbuf = txn.get(image_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)        
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_COLOR)
        return image

    def __get_image_num__(self,txn, label_image_key):
        # print('label image key:', label_image_key)
        return int(txn.get(label_image_key.encode()))

    # 从原始图片中得到字符串的截图
    def __get_char_img__(self, txn, char_pos_key):
        char_pos_info = str(txn.get(char_pos_key.encode()).decode())
        char_file_name, char_pos = char_pos_info.split('|')
        char_pos_img_key = 'origin_%s' % char_file_name.split('.')[0]
        char_img = self.__get_image__(txn, char_pos_img_key)
        x1,y1,x2,y2 = np.fromstring(char_pos, dtype=np.uint, sep=',')
        char_img = char_img[y1:y2, x1:x2, ]
        return char_file_name, char_img

    def __get_image_label__(self,txn, index, label):

        char_img_lists = []
        char_clip_img_lists = []

        for item in list(label):
            label_image_key = 'Sample{}_num_samples'.format(item.upper())
            label_image_number = self.__get_image_num__(txn, label_image_key)
            
            # 字符在原图中的坐标，并得到字符在原图中图片
            char_pos_key = 'Sample{}_pos_{}'.format(item.upper(), np.random.randint(label_image_number))
            # print('char pos key:', char_pos_key)
            char_file_name, char_img = self.__get_char_img__(txn,char_pos_key)
            char_img_lists.append(char_img)

            # 做了二值化转换后的已截取的字符图片, 将其转为灰度图， 方便后面transformer识别为二值化图
            char_clip_key = 'clip_{}'.format(char_file_name.split('.')[0])
            char_clip_img = self.__get_image__(txn, char_clip_key)
            char_clip_img = cv2.cvtColor(char_clip_img, cv2.COLOR_BGR2GRAY)
            char_clip_img_lists.append(char_clip_img)


        # 背景图片
        bg_image_numter = self.__get_image_num__(txn, 'bg_num_samples')
        bg_image_key = 'bg_{}'.format(np.random.randint(bg_image_numter))
        bg_image = self.__get_image__(txn, bg_image_key)

        char_img = char_img_lists
        char_clip_img = char_clip_img_lists

        if np.random.randint(2):
            image, bg_image, label = self.transform(char_clip_img, bg_image, label)
        else:
            image, bg_image, label = self.transform(char_img, bg_image, label)

        return image, label

    def __get_empty_img__(self, txn, index):
        label = BLANK_FLAG
        empty_number = self.__get_image_num__(txn, 'clean_num_samples')
        empty_img_key = 'clean_{}'.format(np.random.randint(0, empty_number))
        empty_img = self.__get_image__(txn, empty_img_key)
        if self.transform is not None:
            image, bg_image, label = self.transform(empty_img, None, label)        
        return image, label



    def pull_train_item(self,txn,index):
        label_key = f'train_{index}'
        label = str(txn.get(label_key.encode()).decode())
        # 从原始图片获取, 暂时不用， 因原始图片是包含（）等信息的，没有截取字符串的
        # image, label = self.__get_origin_image_label__(txn,index,label)
        if  np.random.randint(10) == 0:
            image, label = self.__get_empty_img__(txn, index)
        else:
            image, label = self.__get_image_label__(txn, index, label)
        return image, label


    def pull_valid_item(self,txn, index):
        label_key = f'valid_{index}'
        label = str(txn.get(label_key.encode()).decode())
        if  np.random.randint(10) == 0:
            image, label = self.__get_empty_img__(txn, index)
        else:
            image, label = self.__get_image_label__(txn, index, label)
        return image, label


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    import time
    import os
    import shutil
    from lmdb_transform import CharImgTransform
    parser = argparse.ArgumentParser(description='ocr dataset')
    parser.add_argument('--root_path', default='D:\\PROJECT_TW\\git\\data\\ocr', type=str)
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr\\lmdb', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='train', type=str)    
    args = parser.parse_args()

    tmp_path = 'D:\\PROJECT_TW\\git\\data\\tmp'
    
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    dataset = lmdbDataset(root=args.data_root, 
                          split='train',
                          transform=CharImgTransform(data_root=args.root_path))

    print('data set len :', len(dataset))

    for idx in range(10):
        # try:
        if idx % 100 == 0:
            print('handle %d' % idx)
        image, label = dataset[idx]
        image = image.astype(np.int)
        print('label:', label, ' image shape:', image.shape)

        cv2.imwrite(os.path.sep.join([tmp_path,f'{label}_{idx}.png']),image)
        # except:
        #     pass
        # plt.imshow(image,'gray')
        # plt.show()
    use_cuda = False

    train_loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        pin_memory=True if use_cuda else False,
        collate_fn=adjustCollate(imgH=32, keep_ratio=True),
        num_workers=0)   

    for images, labels in train_loader:
        print('image shape:', images.size(), ' labels :', labels)


    # start_time = time.time()
    # for _ in range(10):
    #     data_iter = iter(train_loader)    
    #     image, label = next(data_iter)
    #     print('label', label, 'image size:', image.size())
    # print('load iter time ：',(time.time() - start_time))

    
