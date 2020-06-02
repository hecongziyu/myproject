# encoding: utf-8

import random
import sys
import lmdb
import numpy as np
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import cv2

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            if imgbuf is not None:
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf)
                    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                    img = img.astype(np.uint8)
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]

                if self.transform is not None:
                    img = self.transform(img)
                # else:
                #     img = torch.tensor(img,dtype=torch.float)

                label_key = 'label-%09d' % index
                label = str(txn.get(label_key.encode()).decode())
                if self.target_transform is not None:
                    label = self.target_transform(label)
                return (img, label)
            else:
                return (None, None)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img



class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

class adjustCollate(object):
    def __init__(self, imgH=32, keep_ratio=True):
        self.imgH = imgH
        self.keep_ratio = keep_ratio
        self.toTensor = transforms.ToTensor()

    def __resize__(self,image):
        image_resize = image.copy()
        if image.shape[0] != 32:
            percent = float(self.imgH) / image_resize.shape[0]
            image_resize = cv2.resize(image_resize,(0,0), fx=percent, fy=percent, interpolation = cv2.INTER_AREA)
        return image_resize

    def __normalize__(self,img):
        # 注意一定需要astype为np.uint8, to tensor 才会认为这是一张图片
        # img = img.astype(np.uint8)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


    def __call__(self, batch):
        images, labels = zip(*batch)
        images = [self.__resize__(image) for image in images]
        PADDING_CONSTANT = 255
        assert len(set([b.shape[0] for b in images])) == 1
        assert len(set([b.shape[2] for b in images])) == 1
        if len(images) > 0:
            dim0 = images[0].shape[0]
            dim1 = max([b.shape[1] for b in images])
            dim2 = images[0].shape[2]
            images_padding = np.full((len(images), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.uint8)
            for idx, img in enumerate(images_padding):
                img[:,:images[idx].shape[1],:] = images[idx]
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
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels



if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description="OCR data set")
    parser.add_argument('--split',default='train', type=str, help='path of the evaluated model')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr\\lmdb', type=str, help='path of the evaluated model')
    args = parser.parse_args()    

    dataset = lmdbDataset(os.path.sep.join([args.data_root, args.split]))

    for image, label in dataset:
        if image is not None:
            print(image.shape)