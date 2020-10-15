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
from xml.dom.minidom import parse

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


# 提取试卷中图片位置信息
class PaperPicDataset(Dataset):
    def __init__(self, data_dir, window=1200,detect_type='pic',transform=None, target_transform=AnnotationTransform()):
        self.data_dir = data_dir
        Dataset.__init__(self)
        self.env = lmdb.open(
            os.path.sep.join([data_dir,'source','lmdb']),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)        
        self.transform = transform
        self.window=window
        self.target_transform = target_transform
        self.detect_type = detect_type
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))
            self.nSamples = nSamples   
            self.dtype_maps = self.__get_dtype_idx__(txn)

        self.file_train_data, self.file_no_pic_train_data = self.__load_file_data__()

    def __load_file_data__(self):
        img_file_lists = os.listdir(join(self.data_dir, 'taged_image', 'pic','sources'))
        img_file_lists = [x for x in img_file_lists if x.find('.png') != -1]
        train_data = []
        no_pic_train_data = []

        for item in img_file_lists:
            image = cv2.imread(join(self.data_dir,'taged_image', 'pic', 'sources', item), cv2.IMREAD_COLOR)
            file_name = item.rsplit('.')[0]

            if os.path.exists(join(self.data_dir, 'taged_image', 'pic','sources_xml', f'{file_name}.xml')):
                domtree = parse(join(self.data_dir, 'taged_image', 'pic','sources_xml', f'{file_name}.xml'))
                imgdom = domtree.documentElement
                img_name = imgdom.getElementsByTagName('filename')[0].firstChild.nodeValue
                box_lists = []
                box_nodes = imgdom.getElementsByTagName('bndbox')
                for box in box_nodes:
                    xmin = int(int(box.getElementsByTagName('xmin')[0].firstChild.nodeValue))
                    ymin = int(int(box.getElementsByTagName('ymin')[0].firstChild.nodeValue))
                    xmax = int(int(box.getElementsByTagName('xmax')[0].firstChild.nodeValue))
                    ymax = int(int(box.getElementsByTagName('ymax')[0].firstChild.nodeValue))
                    box_lists.append([xmin, ymin, xmax, ymax, 0])
                train_data.append((image, np.array(box_lists)))

            else:
                no_pic_train_data.append((image, np.array([[-1,-1,-1,-1,-1]])))
        return train_data,no_pic_train_data


    def __get_dtype_idx__(self,txn):
        dtype_maps = {}
        pic_idx_lists = []

        for idx in range(self.nSamples):
            if idx == 0:
                idx = 1
            target_key = f'pos_{idx}'
            target =  np.frombuffer(txn.get(target_key.encode()), dtype=np.float)
            target = target.astype(np.int)
            target = target.reshape(-1,5)

            if len(np.where(target[:,4]==1)[0]) > 0:
                pic_idx_lists.append(idx)

        dtype_maps['pic'] = pic_idx_lists
        return dtype_maps
            # target = self.__get_target__(txn, f'pos_{index}')


    def __len__(self):
        number = len(self.dtype_maps[self.detect_type]) + len(self.file_train_data) + len(self.file_no_pic_train_data)
        # return len(self.dtype_maps[self.detect_type]) + int(len(self.dtype_maps['pic']) * 0.2)
        return number
        

    def __getitem__(self, index):
        if np.random.randint(3) == 1:
            return self.__get_lmdb_item__(index)
        else:
            return self.__get_file_item__(index)

    def __get_file_item__(self, index):
        if np.random.randint(3) == 1:
            _file_index = np.random.randint(0, len(self.file_no_pic_train_data))
            image, target = self.file_no_pic_train_data[_file_index]
        else:
            _file_index = np.random.randint(0, len(self.file_train_data))
            image, target = self.file_train_data[_file_index]

        boxes,labels= target[:, 0:4], target[:, 4]
        image = image.astype(np.uint8)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes = self.target_transform(boxes, self.window, self.window)
        boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return image, boxes



    def __get_lmdb_item__(self, index):
        if index < len(self.dtype_maps['pic']):
            index = self.dtype_maps[self.detect_type][index]
        else:
            _d_type_idx = np.random.randint(0, len(self.dtype_maps['pic']))
            index = self.dtype_maps['pic'][_d_type_idx]
            # print('real  index:', index)
        with self.env.begin(write=False) as txn:
            image, target = self.pull_train_item(txn, index)
        boxes = target[:, 0:4]
        labels = target[:, 4]
        image = image.astype(np.uint8)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)        
        return image

    def __get_target__(self, txn, target_key):
        target =  np.frombuffer(txn.get(target_key.encode()), dtype=np.float)
        target = target.astype(np.int)
        target = target.reshape(-1,5)
        target = target[target[:,4]==1]
        target[:,4] = 0
        
        if target.shape[0] == 0:
            target = np.array([[-1,-1,-1,-1,-1]])
        return target   

    def pull_train_item(self, txn, index):
        image = self.__get_image__(txn, f'img_{index}')
        target = self.__get_target__(txn, f'pos_{index}')
        return image, target


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    from lmdb_formula_transform import PicTransform
    import cv2
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision import transforms

    transform = transforms.ToTensor()

    def detection_collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                     0 dim
        """
        targets = []
        imgs = []
        # ids = []
        for sample in batch:

            imgs.append(transform(sample[0]))
            targets.append(torch.FloatTensor(sample[1]))
            # ids.append(sample[2])
        return torch.stack(imgs, dim=0), targets

    parser = argparse.ArgumentParser(description='math formula imdb dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\mathdetect', type=str, help='path of the math formula data')
    parser.add_argument('--batch_size',default=16, type=int)

    args = parser.parse_args()

    dataset = PaperPicDataset(data_dir=args.data_root,
                             window=1200,
                             transform=PicTransform(window=1200, max_width=1200, size=512),
                             detect_type='pic',
                             target_transform=AnnotationTransform())
    dataset_size = len(dataset)
    random_sel = np.random.randint(0, len(dataset), 100).tolist()

    for idx in random_sel:
        image, box = dataset[idx]
        print('image  :', image.shape, ' boxes: ', box)
        image = image.astype(np.uint8)
        image = cv2.resize(image, (1200,1200), interpolation=cv2.INTER_AREA)
        # target = target.astype(np.int)
        for p_idx, pos in enumerate(box):
            # pos = pos.astype(np.int)
            x0, y0, x1, y1, label= pos
            x0 = int(1200*x0)
            x1 = int(1200*x1)
            y0 = int(1200*y0)
            y1 = int(1200*y1)

            print(x0, ':', x1, ':', y0, ':', y1, ' label:', label)
            if int(label) == 0:
                cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 2)
            elif int(label) == 1:
                cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 255), 2)
            else:
                pass
        cv2.imwrite(r'D:\PROJECT_TW\git\data\mathdetect\temp\t_' + f'{idx}.png', image)
        # plt.imshow(image)
        # plt.show() 
