"""
Author: Parag Mali
Data reader for the GTDB dataset
Uses sliding windows to generate sub-images

处理包括数学公式、几何图形的数据集
"""
import os
import sys
import torch
import torch.utils.data as data
import cv2
from init import init_args
import numpy as np
from gtdb import box_utils
from gtdb import feature_extractor
import copy
import lmdb
import six
from ocr_pos_transform import GTDBTransform
from torchvision import transforms

transform = transforms.ToTensor()

GTDB_CLASSES = (  # always index 0 is background
   'alpha')

# GTDB_ROOT = osp.join(HOME, "data/GTDB/")

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



class GTDBAnnotationTransform(object):
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
        # print(target)
        for box in target:
            res.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height, box[4]])
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class GTDBDetection(data.Dataset):
    """GTDB Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to GTDB folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name: `GTDB`
    """

    def __init__(self, args, data_file, split='train',
                 transform=None, target_transform=None,
                 dataset_name='GTDB'):

        #split can be train, validate or test
        self.env = lmdb.open(
            data_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.data_file = data_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.window = args.window

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            if self.split == 'train':
                nSamples = int(txn.get('train_num_samples'.encode()))
            else:
                nSamples = int(txn.get('valid_num_samples'.encode()))
            self.nSamples = nSamples
                


    def __getitem__(self, index):
        im, boxes = self.pull_item(index)
        # print('im :', im.shape)
        return im, boxes

    def __len__(self):
        return self.nSamples
        # return 100


    def pull_item(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            if self.split == 'train':
                images, boxes, bg_img = self.pull_train_item(txn, index)
            else:
                images, boxes, bg_img = self.pull_valid_item(txn, index)

        # 打上标注，注明该位置保存有信息
        
        if len(boxes) > 0:
            [x.append(0) for x in boxes]
        else:
            boxes.append(-1.,-1.,-1.,-1.,-1.)
       
        if self.target_transform is not None:
            boxes = self.target_transform(boxes, self.window, self.window)

        boxes = np.array(boxes, dtype=np.float32)

        image = images[0]

        if self.transform is not None:
            image, boxes, bg_img = self.transform(images, boxes, bg_img)


        return image, boxes

    def pull_train_item(self,txn,index):
        label_key = f'train_{index}'
        label = str(txn.get(label_key.encode()).decode())
        random_select = np.random.randint(5)
        if random_select < 2:
            images, boxes, bg_img = self.__get_origin_image_label__(txn, index, label)
        elif random_select < 4:
            images, boxes, bg_img = self.__get_clip_image_label__(txn, index, label)
        else:
            images, boxes, bg_img = self.__get_empty_img__(txn, index)
        return images, boxes, bg_img

    def pull_valid_item(self,txn, index):
        label_key = f'valid_{index}'
        label = str(txn.get(label_key.encode()).decode())
        random_select = np.random.randint(5)
        if random_select < 2:
            images, boxes, bg_img = self.__get_origin_image_label__(txn, index, label)
        elif random_select < 4:
            images, boxes, bg_img = self.__get_clip_image_label__(txn, index, label)
        else:
            images, boxes, bg_img = self.__get_empty_img__(txn, index)
        return images, boxes, bg_img


    def __get_empty_img__(self, txn, index):
        images = []
        boxes = []

        empty_number = self.__get_image_num__(txn, 'clean_num_samples')
        empty_img_key = 'clean_{}'.format(np.random.randint(0, empty_number))
        empty_img = self.__get_image__(txn, empty_img_key)
        empty_boxes = [-1,-1,-1,-1,-1]
        images.append(empty_img)
        boxes.append(empty_boxes)
        return images, boxes, None


    # 从原始图片中得到字符串的截图
    def __get_char_img__(self, txn, char_pos_key, need_clip=False):
        char_pos_info = str(txn.get(char_pos_key.encode()).decode())
        char_file_name, char_pos = char_pos_info.split('|')
        char_pos_img_key = 'origin_%s' % char_file_name.split('.')[0]
        char_img = self.__get_image__(txn, char_pos_img_key)
        x1,y1,x2,y2 = np.fromstring(char_pos, dtype=np.uint, sep=',')
        char_pos = [x1,y1,x2,y2]
        if need_clip:
            char_img = char_img[y1:y2, x1:x2, ]
            char_pos = [0,0,x2-x1, y2-y1]
        return char_file_name, char_img, char_pos

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

    # 直接得到带位置信息的原图, 注意label 位数只能唯一
    def __get_origin_image_label__(self,txn, index, label):

        images = []
        boxes = []
        bg_img = None

        for item in list(label)[0]:
            label_image_key = 'Sample{}_num_samples'.format(item.upper())
            label_image_number = self.__get_image_num__(txn, label_image_key)
            
            # 字符在原图中的坐标，并得到字符在原图中图片
            char_pos_key = 'Sample{}_pos_{}'.format(item.upper(), np.random.randint(label_image_number))
            char_file_name, char_img, char_pos = self.__get_char_img__(txn,char_pos_key)
            images.append(char_img)
            boxes.append(char_pos)

        return images, boxes, bg_img

    # 得到切割后的图片和相应的随机背景图片
    def __get_clip_image_label__(self,txn, index, label):
        # 背景图片
        bg_image_numter = self.__get_image_num__(txn, 'bg_num_samples')
        bg_image_key = 'bg_{}'.format(np.random.randint(bg_image_numter))
        bg_image = self.__get_image__(txn, bg_image_key)

        # 从原始图片中切割的图片，没有做二值化处理
        clip_img_lists = []
        # 从原始图片中切割的图片，已做二值化处理
        clip_img_bit_lists = []

        for item in list(label):
            label_image_key = 'Sample{}_num_samples'.format(item.upper())
            label_image_number = self.__get_image_num__(txn, label_image_key)
            # 字符在原图中的坐标，并得到字符在原图中图片
            char_pos_key = 'Sample{}_pos_{}'.format(item.upper(), np.random.randint(label_image_number))
            # print('char pos key:', char_pos_key)
            char_file_name, char_img, char_pos = self.__get_char_img__(txn,char_pos_key,need_clip=True)
            clip_img_lists.append([char_img,0])

            # 做了二进制转换后的已截取的字符图片
            char_clip_key = 'clip_{}'.format(char_file_name.split('.')[0])
            char_clip_img = self.__get_image__(txn, char_clip_key)
            clip_img_bit_lists.append([char_clip_img,1])


        images = None
        boxes = None

        if np.random.randint(2):
            images = clip_img_lists
        else:
            images = clip_img_bit_lists

        boxes = [[0,0,x[0].shape[1], x[0].shape[0]] for x in images]

        return images, boxes, bg_image




if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    import time
    import os
    import shutil
    import torch.utils.data as data
    import sys
    # parser = argparse.ArgumentParser(description='ocr dataset')
    # parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr\\lmdb', type=str, help='path of the evaluated model')
    # parser.add_argument('--split', default='train', type=str)    
    # parser.add_argument('--window', default=1200,type=int)   
    # args = parser.parse_args()
    args = init_args()
    print('args:', args)
    tmp_path = 'D:\\PROJECT_TW\\git\\data\\tmp'
    
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)


    # 注意GTDBAnnotationTransform  [box[0]/width, box[1]/height, box[2]/width, box[3]/height, 0]

    dataset = GTDBDetection(args, data_file=args.data_root, split='train',
                            transform = GTDBTransform(data_root=args.root_path, window=args.window, size=args.size),
                            target_transform = GTDBAnnotationTransform()
                            )    


    for index in range(2000):
        if index % 100 == 0:
            print('handle {} image '.format(index))
        im, boxes = dataset[index]
        im = im.astype(np.uint8)
        heigh, width,_ = im.shape
        for box in boxes:
            x0,y0,x1,y1,target = box
            b_width = x1-x0
            b_height= y1-y0
            # 注意 下面方式是对坐标进行了转换的计算, 后续需修改成下面方式
            # print(int(x0*width),int(y0*heigh), int(b_width*width), int(b_height*heigh))
            # print('boxes :', int(x0*width),int(y0*heigh), int(b_width*width), int(b_height*heigh), ',target:',target)
            if target != -1:
                cv2.rectangle(im, (int(x0*width),int(y0*heigh), int(b_width*width), int(b_height*heigh)), (0, 0, 255), 1)
        
        # plt.imshow(im)        
        # plt.show()
        cv2.imwrite(os.path.sep.join([tmp_path,f'{index}.png']),im)



    # data_loader = data.DataLoader(dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)   

    # for imgs, targets in data_loader:
    #     print('image size :', imgs) 
    #     sys.exit(0)   
        # print('targets :',targets)
        # print('labels :', labels)

