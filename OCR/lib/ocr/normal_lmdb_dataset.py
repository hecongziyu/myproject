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



'''
https://palikar.github.io/posts/pytorch_datasplit/  切割数据
https://blog.csdn.net/hao5335156/article/details/80593349?utm_source=blogxgwz7 关于图像 np.uint8 和np.float32的区别
        # 原image 训练数据数据类型为np.uint8, 不能改成np.float32, 改成np.float32数据发生较大的变化
        # 如array([[300., 400., 300., 300., 100.]]) 改成 np.uint8 变为  array([[ 44, 144,  44,  44, 100]], dtype=uint8)
        # np.uint8 取值范围为 0, 255
        # image = image.astype(np.float32)
        # 注意当不做转换时, image transform 后数值为为 1, 当做image astype(np.float32)后，数据还是255.

注意：需要将图片大小变成相同。
处理方式一：
1、过滤掉与第一个图片大小不相同的图片. (注意采用该方式时最好先将图片大小进行排序)

处理方式二：
1、找到该批次图片高度最大图片，将其它图片按比例缩放到该高度。可考虑设置缺省高度为64
2、找到该批次图片宽度最大图片，其它图片填充空白区域到相同
'''
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



def expand_width(image, imgH, max_width):
    image_ext = np.ones((imgH, max_width, image.shape[2]),dtype=image.dtype) * 255
    image_ext[0:image.shape[0], 0:image.shape[1],:] = image
    return image_ext






class lmdbDataset(Dataset):
    def __init__(self, root=None,split='train', transform=None, max_len=150):
        self.env = lmdb.open(
            os.path.sep.join([root,'lmdb']),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.transform = transform
        self.split = split
        self.max_len = max_len

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))
            self.nSamples = nSamples


    def __len__(self):
        if self.split == 'train':
            return (self.nSamples + (self.nSamples // 2)) * 100
        else:
            return (self.nSamples + (self.nSamples // 2)) * 100 // 10

    def __getitem__(self, index):
        index = (index % (self.nSamples + self.nSamples // 2))
        # assert index <= len(self), 'index range error'
        if index < self.nSamples:
            with self.env.begin(write=False) as txn:
                image, target = self.pull_train_item(txn, index)
                print('real img :', image.shape)
        else:
            image = None
            target = 'z'
        # target = " ".join(target.split()[0:self.max_len])
        return image, target

    def __get_image__(self, txn, image_key):
        imgbuf = txn.get(image_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)        
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_COLOR)
        return image

    def __get_target__(self, txn, target_key):
        return str(txn.get(target_key.encode()).decode())

    def pull_train_item(self, txn, index):
        image = self.__get_image__(txn, f'i_{index}')
        target = self.__get_target__(txn, f't_{index}')
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


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
    parser.add_argument('--data_root',default=r'D:\\PROJECT_TW\\git\\data\\ocr\\number', type=str, help='path of the evaluated model')
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

    dataset = lmdbDataset(root=args.data_root, split='valid',transform=ImgTransform(data_root=args.data_root))

    dataset_size = len(dataset)

    print('dataset size:', dataset_size)
    # random_sel = np.random.randint(0 , len(dataset), 50).tolist()

    for ridx, idx in  enumerate(list(range(50))):
        image, target = dataset[idx]
        image = image.astype(np.uint8)
        print(idx , '--->', ' image shape:', image.shape)
        print('target:', target.strip())
        cv2.imwrite(os.path.sep.join([args.data_root,'valid_img',f'{idx}_{target}.png']),image)
        
        # print('target:', [vocab.sign2id(x,3) for x in target.strip()])
        # plt.imshow(image)
        # plt.show()


    # train_loader = DataLoader(dataset,
    #                     batch_size=args.batch_size,
    #                     shuffle=True,
    #                     pin_memory=True if use_cuda else False,
    #                     collate_fn=adjustCollate(imgH=32, keep_ratio=True),
    #                     num_workers=0)         


    # for _ in range(2):
    #     example_batch = next(iter(train_loader)) 

    #     print('image size: ', example_batch[0].size())
    #     print('label :', example_batch[1])
        # break;

        # concatenated = torch.cat((example_batch[0],example_batch[1]),0) 
        # imshow(torchvision.utils.make_grid(example_batch[0], nrow=args.batch_size))
