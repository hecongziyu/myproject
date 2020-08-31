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
from torchvision import transforms

transform = transforms.ToTensor()

def collate_fn(batch):
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
    img_0_lists = []
    img_1_lists = []
    # ids = []
    for sample in batch:

        img_0_lists.append(transform(sample[0]))
        img_1_lists.append(transform(sample[1]))
        targets.append(sample[2])

    targets = torch.tensor(targets, dtype=torch.float)
    targets = targets.unsqueeze(1)

    return torch.stack(img_0_lists, dim=0),torch.stack(img_1_lists, dim=0), targets

class LmdbDataset(Dataset):
    def __init__(self, data_dir=None,split='train', transform=None):
        self.transform = transform
        self.env = lmdb.open(
            os.path.sep.join([data_dir,'lmdb']),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))
            self.nSamples = nSamples


    def __len__(self):
        return self.nSamples
        # return 30


    def __get_image__(self, txn, image_key):
        imgbuf = txn.get(image_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)        
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8),cv2.IMREAD_COLOR)
        return image


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_key = f'img_{index}'
        with self.env.begin(write=False) as txn:
            image_0 = self.__get_image__(txn,image_key)

            label = np.random.randint(0,2)

            if label == 0:
                image_1 = self.__get_image__(txn,image_key)
            else:
                _idx_lists = [x for x in range(len(self)) if x != index]
                _idx = _idx_lists[np.random.randint(0, len(_idx_lists))]
                # print('_idx :', f'img_{_idx}')
                image_1 = self.__get_image__(txn,f'img_{_idx}')

        if self.transform:
            # print('image 0:', image_0.shape)
            image_0 = self.transform(image_0)
            # print('image 1:', image_1.shape)
            image_1 = self.transform(image_1)


        return image_0, image_1, label       

if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    from sia_transform import LmdbTransform
    import torchvision
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\siamese',help='data set root')
    parser.add_argument('--batch_size', default=8, type=int,help='data set root')
    args = parser.parse_args()    

    def imshow(img,text=None,should_save=False): 
        #展示一幅tensor图像，输入是(C,H,W)
        npimg = img.numpy() #将tensor转为ndarray
        npimg = npimg.astype(np.uint8)
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
        plt.show()        

    dataset = LmdbDataset(data_dir=args.data_root, split='train', transform=LmdbTransform())
    print('data size :', len(dataset))

    # for idx in range(5):
    #     image_0, image_1, label = dataset[idx]
    #     image_0 = image_0.astype(np.uint8)
    #     plt.imshow(image_0)
    #     plt.show()
    #     print('label :', label, ' image_0 shape:', image_0.shape, ' image_1 shape:', image_1.shape)

    # dataloader = DataLoader(siamese_dataset,shuffle=True,batch_size=4)

    rsample = torch.utils.data.RandomSampler(data_source=list(range(len(dataset))), replacement=True)


    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                            sampler=rsample,
                            collate_fn=collate_fn,
                            pin_memory=False)

    for _ in range(2):
        example_batch = next(iter(dataloader)) 

        print('image size: ', example_batch[0].size())
        break;

        concatenated = torch.cat((example_batch[0],example_batch[1]),0) 

        print(concatenated.size())
        print(example_batch[2].numpy())

        imshow(torchvision.utils.make_grid(concatenated, nrow=args.batch_size))

    
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0) 
                            # sampler=valid_sampler)