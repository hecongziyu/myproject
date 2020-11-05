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
from torchvision import transforms
from os.path import join


transform = transforms.ToTensor()


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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

    # target input is x0,y0,x1,y1 convert to x,y,w,h and norm 
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotations. This will be the list of bounding boxes
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        
        # res = []
        target = xyxy2xywh(np.array(target,dtype=np.float32))
        target[:,[0,2]] = target[:,[0,2]] / width
        target[:,[1,3]] = target[:,[1,3]] / height

        # read the annotations
        # for box in target:
            # res.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height])
        return target.tolist()  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def collate_fn(batch):
    # img, label, path, shapes = zip(*batch)  # transposed
    # for i, l in enumerate(label):
        # l[:, 0] = i  # add target image index for build_targets()
    # return torch.stack(img, 0), torch.cat(label, 0), path, shapes        
    img, label = zip(*batch)  # transposed
    # print('img type:', type(img),':',len(img) ,' label :', label)

    # BGR to RGB, to 3x416x416
    img = [transform(x).float() for x in img]
    # print('collate fn before label:', label)
    label = [torch.from_numpy(x).float() for x in label]


    # print('collate fn after label:', label)
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)        


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
            if height > 2048:
                radio = 2048/height
            else:
                radio = 1

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
        # return len(self.ext_train_data)
        # return 1
        # return self.nSamples * 50
        return (self.nSamples + len(self.ext_train_data)) * 50

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

        labels = np.hstack((np.expand_dims(labels, axis=1), boxes))

        # 去掉非目标区域
        sel_labels = np.where(labels[:,0] != -1)
        labels = labels[sel_labels]


        # boxes = boxes[sel_labels]


        # 增加一列，用于yolo训练时作为图片索引, 注意图片索引是根据batch idx设置，这里只扩展一位

        # print('labels shape:', labels.shape, ' boxes len :', len(boxes))
        labels = np.hstack((np.zeros((len(boxes),1)),  labels))

        if labels.shape[0] == 0:
            raise Exception('训练数据没有目标检测数据 %s ！' % index)       


        return image, labels

        

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
        # print('target :', target)
        target = target[np.where(target[:,4] != -1)]
        if target.shape[0] == 0:
            # target = np.array([[-1,-1,-1,-1,-1]])
            target = np.array([])
        return target   


    def pull_train_item(self, txn, index):
        index = index % (self.nSamples + len(self.ext_train_data))
        image = None
        target = np.array([])
        if index < self.nSamples:
            image = self.__get_image__(txn, f'bg_{index}')
            target = self.__get_target__(txn, f'target_{index}')
            # print('target :', target, ' target shape:', target.shape)
            # if target.shape[0] > 0:
                # target = target[np.where(target[:,4]) != -1]

        # yolo 训练需确保数据都包含检测目标
        if target.shape[0] == 0:
            ext_idx = np.random.randint(len(self.ext_train_data))
            _image, _target = self.ext_train_data[ext_idx]
            image = _image.copy()
            target = _target.copy()

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
    from qrtransform import QRTransform
    from os.path import join
    parser = argparse.ArgumentParser(description='math formula imdb dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\qrdetect', type=str, help='path of the math formula data')
    parser.add_argument('--batch_size',default=16, type=int)
    args = parser.parse_args()

    transform = transforms.ToTensor()

    dataset = QRDataset(data_dir=args.data_root,
                        window=416, transform=QRTransform(window=416), target_transform=AnnotationTransform())    

    print('data set len :', len(dataset))
    random_sel = np.random.randint(0, len(dataset), 2000).tolist()

    # len(dataset)-50,
    for ridx, idx in enumerate(random_sel):
        image, labels = dataset[idx]
        image = image.astype(np.uint8)
        image = cv2.resize(image, (416,416), interpolation=cv2.INTER_AREA)
        print('gen ridx %s 数据 ' % ridx)    
        # print('image  :', image.shape, ' boxes: ', labels)
        image = image.astype(np.uint8)
        for box in labels:
            imgidx,label,x, y, w, h = box
            # print('qr img size :', w*h * 416)
            # print(x,',',y, ',', w, ',', h)
            # print('w : %s h: %s ' % (w*416, h*416))

            x0,y0,x1,y1 = xywh2xyxy(np.array([[x,y,w,h]], dtype=np.float32)).tolist()[0]
            # print(x0,',',y0, ',', x1, ',', y1)

            x0 = int(416*x0) 
            x1 = int(416*x1)  
            y0 = int(416*y0) 
            y1 = int(416*y1)     

            # print(x0,',',y0, ',', x1, ',', y1)

            if int(label) == 0:
                cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 2)

        cv2.imwrite(join(args.data_root,'valid_imgs', f'{ridx}_.png'), image)

        # plt.imshow(image)
        # plt.show()        

    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=3,
    #                                          shuffle=False,  # Shuffle=True unless rectangular training is used
    #                                          pin_memory=True,
    #                                          collate_fn=collate_fn)    

    # for _ in range(3):
    #     for idx, (imgs, labels) in enumerate(dataloader):
    #         print('idx :', idx, ' imags shape:', imgs.size(), ' labels shape:', labels.size())
    #         print('images: --> \n', imgs)
    #         print('labels ---> \n ', labels)
    #         break