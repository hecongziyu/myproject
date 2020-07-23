# encoding: utf-8
import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import six
import torch

'''
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
            # if self.split == 'train':
            #     nSamples = int(txn.get('train_num_samples'.encode()))
            # else:
            #     nSamples = int(txn.get('valid_num_samples'.encode()))
            nSamples = int(txn.get('total'.encode()))
            self.nSamples = nSamples



    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            image, target = self.pull_train_item(txn, index)
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
    parser = argparse.ArgumentParser(description='latex imdb dataset')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\im2latex\\lmdb', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='train', type=str)    
    args = parser.parse_args()

    ds = lmdbDataset(root=args.data_root)

    print('ds len:', len(ds))

    random_sel = np.random.randint(0, len(ds), 5).tolist()

    for idx in random_sel:
        image, target = ds[idx]
        image = image.astype(np.int)
        print('target:', target.strip())
        plt.imshow(image)
        plt.show()
    #     