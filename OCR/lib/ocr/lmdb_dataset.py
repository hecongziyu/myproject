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
from torch.utils.data import DataLoader

class resizeNormalize(object):
    def __init__(self, size):
        self.size = size
        self.toTensor = transforms.ToTensor()
    def __call__(self, image):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


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
        return self.nSamples
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
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_GRAYSCALE)
        return image

    def __get_image_num__(self,txn, label_image_key):
        # print('label image key:', label_image_key)
        return int(txn.get(label_image_key.encode()))


    def pull_train_item(self,txn,index):
        label_key = f'train_{index}'
        label = str(txn.get(label_key.encode()).decode())
        label_image_key = 'Sample{}_num_samples'.format(label.upper())
        label_image_number = self.__get_image_num__(txn, label_image_key)
        image_key = 'Sample{}_{}'.format(label.upper(), np.random.randint(label_image_number))
        image = self.__get_image__(txn, image_key)
        return image, label


    def pull_valid_item(self,txn, index):
        pass


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    import time
    parser = argparse.ArgumentParser(description='ocr dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr\\lmdb', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='train', type=str)    
    args = parser.parse_args()

    dataset = lmdbDataset(root=args.data_root, split=args.split)

    print(len(dataset))

    # for idx in range(10):
    #     image, label = dataset[idx]
    #     print('label:', label, ' image shape:', image.shape)
    #     # plt.imshow(image,'gray')
        # plt.show()
    use_cuda = False

    train_loader = DataLoader(
        dataset,
        batch_size=10,
        pin_memory=True if use_cuda else False,
        collate_fn=alignCollate(imgH=32, imgW=256, keep_ratio=True),
        num_workers=0)   

    start_time = time.time()
    for _ in range(10):
        data_iter = iter(train_loader)    
        image, label = next(data_iter)
        print('label', label, 'image size:', image.size())
    print('load iter time ：',(time.time() - start_time))

    
