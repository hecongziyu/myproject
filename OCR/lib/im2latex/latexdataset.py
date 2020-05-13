import torch.utils.data as data
import os
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
import torch
from build_vocab import PAD_TOKEN, UNK_TOKEN
import numpy as np

MEANS = (246, 246, 246)

'''
注意：需要将图片大小变成相同。
处理方式一：
1、过滤掉与第一个图片大小不相同的图片. (注意采用该方式时最好先将图片大小进行排序)

处理方式二：
1、找到该批次图片高度最大图片，将其它图片按比例缩放到该高度。可考虑设置缺省高度为64
2、找到该批次图片宽度最大图片，其它图片填充空白区域到相同
'''
def collate_fn(sign2id,batch,max_img_width=1200):
    transform = transforms.ToTensor()
    size = batch[0][0].shape
    batch = [img_formula for img_formula in batch if img_formula[0].shape[1] < max_img_width]
    # print(len(batch))

    # for imt in batch:
    #     print('origin image size: ', imt[0].shape)

    # image_widths = [x[0].shape[1] for x in batch]
    # max_width = max(image_widths)

    # print('max width :', max_width)
    # batch = [img_formula for img_formula in batch if img_formula[0].shape == size]
    # sort by the length of formula
    batch.sort(key=lambda img_formula: len(img_formula[1].split()), reverse=True)
    imgs, formulas = zip(*batch)
    imgs = [expand_width(x, size[0], max_img_width) for x in imgs]    
    formulas = [formula.split() for formula in formulas]
    # targets for training , begin with START_TOKEN
    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    # targets for calculating loss , end with END_TOKEN
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




class LatexDataset(data.Dataset):
    """GTDB Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to GTDB folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
        (eg: take in caption string, return tensor of word indices)
        dataset_name: `GTDB`
    """	
    def __init__(self, args, data_file, split='train', transform=None, target_transform=None, max_len=512,dataset_name='latex'):
        self.root = args.dataset_root
        self.data_file = data_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.nSamples = self.get_data_content()
        self.max_len = max_len
        self.images = {}
        self.latex_text = {}        
        self.read_all_image()
        self.read_all_latex()
        # print(self.latex_text)

    def read_all_image(self):
        for idx, (image_file, latex_id) in enumerate(self.nSamples):
            self.images[idx] = cv2.imread(os.path.sep.join([self.root,"gen_images",image_file]), cv2.IMREAD_COLOR)

    def read_all_latex(self):
        with open(os.path.sep.join([self.root, self.data_file]), 'r') as f:
            latex_lists = f.readlines()
        for idx, (image_file, latex_id) in enumerate(self.nSamples):
            self.latex_text[idx] = latex_lists[int(latex_id)].replace('\n','')

    def get_data_content(self):
        '''
            content: image name, latex line
        '''
        file_name = os.path.sep.join([self.root, 'latex_{}_filter.txt'.format(self.split)])
        with open(file_name, 'r') as f:
            contents = f.readlines()
        contents=[x.replace('\n','').split() for x in contents]
        return contents

    def __len__(self):
        return len(self.nSamples)


    def __getitem__(self, index):
        image, latex = self.pull_item(index)
        return image, latex


    def pull_item(self,index):
        image = self.images[index]
        latex = " ".join(self.latex_text[index].split()[0:self.max_len])
        if self.transform is not None:
            image = self.transform(image)

        # if self.target_transform is not None:
        #     target4traing, target4loss = self.target_transform(latex)

        return image, latex


if __name__ == '__main__':
    import argparse
    from latextransform import LatexImgTransform
    from matplotlib import pyplot as plt
    from build_vocab import Vocab, load_vocab


    MEANS = (246, 246, 246)
    parser = argparse.ArgumentParser(description='latex dataset')
    parser.add_argument('--dataset_root', default='D:\\PROJECT_TW\\git\\data\\im2latex',
                        help='data set root')
    parser.add_argument('--data_file', default='latex_formul_normal.txt',
                        help='data set root')
    args = parser.parse_args()

    vocab = load_vocab(args.dataset_root)
    latex_ds = LatexDataset(args, data_file=args.data_file,split='test',
                            transform=LatexImgTransform(imgH=256, mean=MEANS,data_root=args.dataset_root),
                            target_transform=None,max_len=512)

    for index in range(len(latex_ds)):
        image, latex = latex_ds[index]
        print('latex :', latex, ' image size:', image.shape, '\n')
        print('image: \n', image)
        plt.imshow(image)
        plt.show()
    # use_cuda = False

    # data_loader = DataLoader(latex_ds,
    #     batch_size=5,
    #     collate_fn=partial(collate_fn, vocab.sign2id),
    #     pin_memory=True if use_cuda else False,
    #     num_workers=1)

    # for imgs, tgt4training, tgt4cal_loss in data_loader:
    #     print(imgs.size(), tgt4training.size(), tgt4cal_loss.size())