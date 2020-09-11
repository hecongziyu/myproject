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
from xml.dom.minidom import parse
import logging

class AnnotationTransform(object):
    """Transforms a GTDB annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None):
        pass
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotations. This will be the list of bounding boxes
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        
        res = []
        # read the annotations
        for box in target:
            res.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height])
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class QRDataset(Dataset):
    '''
    试卷定点识别， 训练数据来源
    1）初始化导入的训练数据，保存在lmdb数据库存中
    2）识别错误的数据， 保存在文件中
    训练时需两部分数据结合一起进行训练
    '''
    def __init__(self, data_dir, window=1200,transform=None, target_transform=None):
        Dataset.__init__(self)
        self.env = lmdb.open(
            os.path.sep.join([data_dir,'lmdb']),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)        
        self.transform = transform
        self.target_transform = target_transform
        self.window=window
        self.data_dir = data_dir
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('bg_total'.encode()))
            qrNsamples = int(txn.get('qr_total'.encode()))
            self.nSamples = nSamples   
            self.qrNsamples = qrNsamples

        self.ext_train_data = self.__load_file_data__()
        logging.info('lmdb data len : %s  file data len : %s' % (self.nSamples, len(self.ext_train_data)))

    def __load_file_data__(self):
        '''
        加载识别错误图片重新进入训练
        '''
        file_lists = os.listdir(join(self.data_dir, 'error_imgs', 'taged'))
        file_lists = [x for x in file_lists if x.find('.xml') != -1]

        train_data = []

        for item in file_lists:
            domtree = parse(join(self.data_dir,'error_imgs', 'taged',item))
            imgdom = domtree.documentElement
            img_name = imgdom.getElementsByTagName('filename')[0].firstChild.nodeValue
            image = cv2.imread(join(self.data_dir,'error_imgs', 'taged', img_name), cv2.IMREAD_COLOR)
            height, width, _ = image.shape
            radio = 2048/height

            image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)

            # image = image.astype(np.uint8)
            box_lists = []
            box_nodes = imgdom.getElementsByTagName('bndbox')
            for box in box_nodes:
                xmin = int(int(box.getElementsByTagName('xmin')[0].firstChild.nodeValue) * radio)
                ymin = int(int(box.getElementsByTagName('ymin')[0].firstChild.nodeValue) * radio)
                xmax = int(int(box.getElementsByTagName('xmax')[0].firstChild.nodeValue) * radio)
                ymax = int(int(box.getElementsByTagName('ymax')[0].firstChild.nodeValue) * radio)
                box_lists.append([xmin, ymin, xmax, ymax, 0])

            train_data.append((image, np.array(box_lists)))
        return train_data

    def __len__(self):
        # return self.nSamples * 50
        return self.nSamples + len(self.ext_train_data)

    def __getitem__(self, index):

        # rand_area = [x for x in range(self.nSamples + len(self.ext_train_data)) if x not in [907,148,493,505,679,689,784,813,865]]
        # index = rand_area[np.random.randint(len(rand_area))]

        with self.env.begin(write=False) as txn:
           qrImage, image, target = self.pull_train_item(txn, index)

        image = image.astype(np.uint8)


        if self.transform:
            # print('qr image shape :', qrImage.shape, ' image shape :', image.shape, 'target :', target.shape)
            image, qrImage, target = self.transform(image, qrImage, target)

        boxes = target[:, 0:4]
        labels = target[:, 4]        
        #     image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes = self.target_transform(boxes, self.window, self.window)

        boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return image, boxes

        

    def __get_image__(self, txn, image_key):
        imgbuf = txn.get(image_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)        
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)        
        return image

    def __get_target__(self, txn, target_key):
        target =  np.frombuffer(txn.get(target_key.encode()), dtype=np.float)
        target = target.astype(np.int)
        target = target.reshape(-1,5)
        if target.shape[0] == 0:
            target = np.array([[-1,-1,-1,-1,-1]])

        return target   


    def pull_train_item(self, txn, index):
        if index < self.nSamples:
            image = self.__get_image__(txn, f'bg_{index}')
            target = self.__get_target__(txn, f'target_{index}')
        else:
            logging.info('index %s load data from ext train data .')
            image, target = self.ext_train_data[index - self.nSamples]
        # qr_idx = np.random.randint(0, self.qrNsamples)
        qr_idx = 0
        qrImage = self.__get_image__(txn, f'qr_{qr_idx}')

        return qrImage,image, target

if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    import cv2
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision import transforms
    from qr_lmdb_transform import QRTransform
    from os.path import join
    parser = argparse.ArgumentParser(description='math formula imdb dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\qrdetect', type=str, help='path of the math formula data')
    parser.add_argument('--batch_size',default=16, type=int)
    args = parser.parse_args()

    transform = transforms.ToTensor()

    dataset = QRDataset(data_dir=args.data_root,
                        window=1200, transform=QRTransform(), target_transform=AnnotationTransform())    

    print('data set len :', len(dataset))
    random_sel = np.random.randint(0, len(dataset), 100).tolist()

    # len(dataset)-50,
    for ridx, idx in enumerate(range(len(dataset))):
        image, boxes = dataset[idx]
        image = image.astype(np.uint8)
        image = cv2.resize(image, (1200,1200), interpolation=cv2.INTER_AREA)
                
        print('image  :', image.shape, ' boxes: ', boxes)
        image = image.astype(np.uint8)
        for box in boxes:
            x0, y0, x1, y1, label= box
            x0 = int(1200*x0) - 5
            x1 = int(1200*x1) + 5 
            y0 = int(1200*y0) - 5
            y1 = int(1200*y1) + 5    
            if int(label) == 0:
                cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 2)

        cv2.imwrite(join(args.data_root,'valid_imgs', f'{ridx}_.png'), image)

        # plt.imshow(image)
        # plt.show()        