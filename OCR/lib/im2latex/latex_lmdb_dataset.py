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
from build_vocab import PAD_TOKEN, UNK_TOKEN



'''
https://palikar.github.io/posts/pytorch_datasplit/  切割数据
注意：需要将图片大小变成相同。
处理方式一：
1、过滤掉与第一个图片大小不相同的图片. (注意采用该方式时最好先将图片大小进行排序)

处理方式二：
1、找到该批次图片高度最大图片，将其它图片按比例缩放到该高度。可考虑设置缺省高度为64
2、找到该批次图片宽度最大图片，其它图片填充空白区域到相同
'''
def collate_fn(sign2id,batch,max_img_width=600):
    size = batch[0][0].shape
    # print('size :', size)
    batch = [img_formula for img_formula in batch if img_formula[0].shape == size]
    transform = transforms.ToTensor()
    batch.sort(key=lambda img_formula: len(img_formula[1].split()), reverse=True)

    imgs = []
    formulas = []
    for sample in batch:
        imgs.append(sample[0])
        formulas.append(sample[1])
    formulas = [formula.split() for formula in formulas]
    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    tgt4cal_loss = formulas2tensor(add_end_token(formulas), sign2id)
    imgs = [transform(x) for x in imgs]    
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgt4training, tgt4cal_loss

def expand_width(image, imgH, max_width):
    image_ext = np.ones((imgH, max_width, image.shape[2]),dtype=image.dtype) * 255
    image_ext[0:image.shape[0], 0:image.shape[1],:] = image
    return image_ext



def formulas2tensor(formulas, sign2id):
    """convert formula to tensor"""
    batch_size = len(formulas)
    max_len = len(formulas[0])
    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    for i, formula in enumerate(formulas):
        for j, sign in enumerate(formula):
            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    return tensors


def add_start_token(formulas):
    return [['<s>']+formula for formula in formulas]


def add_end_token(formulas):
    return [formula+['</s>'] for formula in formulas]




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
            # if self.split == 'train':
            #     nSamples = int(txn.get('train_num_samples'.encode()))
            # else:
            #     nSamples = int(txn.get('valid_num_samples'.encode()))
            nSamples = int(txn.get('total'.encode()))
            self.nSamples = nSamples



    def __len__(self):
        # return self.nSamples
        return 100
        # return 5

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            image, target = self.pull_train_item(txn, index)
        target = " ".join(target.split()[0:self.max_len])
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
        return image, target


if __name__ == '__main__':
    import argparse
    from matplotlib import pyplot as plt
    from build_vocab import Vocab, load_vocab
    from functools import partial
    import time
    from torch.utils.data.sampler import SubsetRandomSampler

    parser = argparse.ArgumentParser(description='latex imdb dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\im2latex', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='train', type=str)    
    parser.add_argument('--batch_size', default=16, type=int)    
    parser.add_argument("--cuda", action='store_true',default=True, help="Use cuda or not")   
    parser.add_argument("--max_len", type=int,default=150, help="Max size of formula")     
    parser.add_argument("--shuffle", action='store_true',default=True, help="Use cuda or not")  
    args = parser.parse_args()

    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    # ds = lmdbDataset(root=args.data_root)

    vocab = load_vocab(args.data_root)

    print(vocab.id2sign)

    print('vocab:', len(vocab))

    dataset = lmdbDataset(root=args.data_root, split='train', max_len=args.max_len)

    dataset_size = len(dataset)

    # indices = list(range(dataset_size))

    # val_train_split=0.1

    # valid_split = int(np.floor(val_train_split * dataset_size))

    # if args.shuffle:
    #     np.random.shuffle(indices)    

    # train_indices, valid_indices = indices[valid_split:], indices[:valid_split]
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(valid_indices)

    # train_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,collate_fn=partial(collate_fn, vocab.sign2id), sampler=train_sampler)
    # valid_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,collate_fn=partial(collate_fn, vocab.sign2id), sampler=valid_sampler)

    # print('train loader:', len(train_loader))
    # print('valid loader:', len(valid_loader))

    random_sel = np.random.randint(0, len(dataset), 5).tolist()

    print('random_sel:', random_sel)

    for idx in random_sel:
        image, target = dataset[idx]
        image = image.astype(np.int)
        print('target:', target.strip())
        plt.imshow(image)
        plt.show()

    # dataset = lmdbDataset(root=args.data_root, split='train')
    # print(len(dataset))
    # data_loader = DataLoader(
    #                 lmdbDataset(root=args.data_root, split='train', max_len=args.max_len),
    #                  # transform=LatexImgTransform(imgH=args.image_height, mean=MEANS,data_root=args.dataset_root),
    #                  # max_len=args.max_len),
    #                  shuffle=True,
    #                  batch_size=args.batch_size,
    #                  collate_fn=partial(collate_fn, vocab.sign2id),
    #                  pin_memory=False,
    #                  num_workers=0)

    # print(len(data_loader))
    # count = 0
    # for epoch in range(5):
    #     count = 0
    #     start_time = time.time()
    #     for imgs, tgt4training, tgt4cal_loss in train_loader:
    #         # print('imgs size :', imgs.size())
    #         # print('imgs:', imgs.size())
    #         count = count + 1
    #         if count == 1:
    #             print('imgs size:', imgs.size())
    #             print('tgt4training:', tgt4training)
    #         if count % 1000 == 0:
    #             pass
    #             # print('imgs size:', imgs.size())
    #             # print('tgt4training:', tgt4training)
    #             # print('tgt4cal_loss:', tgt4cal_loss)


    #     print('eooch:', epoch, ' load time:', (time.time() - start_time))