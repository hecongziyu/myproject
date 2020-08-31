import argparse
from build_vocab import Vocab, load_vocab
import torch
from decoding import LatexProducer
# from latextransform import GenResize,ConvertFromInts,BackGround,ExpandWidth
from latex_lmdb_dataset import lmdbDataset,collate_fn
from latex_lmdb_transform import ImgTransform
from model import Im2LatexModel
import cv2
from torchvision import transforms
from latexdataset import LatexDataset,collate_fn,MEANS,expand_width
from torch.utils.data import DataLoader
from latextransform import LatexImgTransform
from functools import partial
from matplotlib import pyplot as plt
import time
import os
import numpy as np



def load_model(model_path):
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model_args = checkpoint['args']    
    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model    

def valid(latex_producer,image_path, data_root):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    resize = GenResize(imgH=128)
    expandWidth = ExpandWidth(128,600)
    convertInts = ConvertFromInts()
    # background = BackGround(data_root)
    # image = resize(image)
    # image = background(image)
    
    # image = expand_width(image, imgH=128, max_width=600)
    image = convertInts(image)
    image = resize(image)
    image = expandWidth(image) 

    showimg = image.copy()
    showimg = showimg.astype(np.int)

    plt.imshow(showimg)
    plt.show()

    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    print('input image size:', image_tensor.size())
    formula = latex_producer(image_tensor)

    return formula

def batch_valid(latex_producer,vocab,args):
    start = time.time()
    val_loader = DataLoader(
        LatexDataset(args, data_file=args.data_file,split='valid', 
                     transform=LatexImgTransform(imgH=args.image_height, mean=MEANS,data_root=args.dataset_root),
                     max_len=args.max_len),
                     shuffle=True,
                     batch_size=1,
                     collate_fn=partial(collate_fn, vocab.sign2id),
                     pin_memory=True if use_cuda else False,
                     num_workers=4)

    for imgs, _, tgt4cal_loss in val_loader:
        try:
            print(' load data time :',  (time.time() - start ))
            start = time.time()
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            results = latex_producer(imgs)
            print('predict  time :',  (time.time() - start ))
            start = time.time()
            print('results:', results)
            print('tgt4cal_loss:', reference)
        except RuntimeError:
            break

def batch_valid_2(latex_producer, vocab, args):
    dataset = lmdbDataset(root=args.dataset_root, split='train', max_len=args.max_len, transform=ImgTransform())
    print('len data set ', len(dataset))
    valid_loader = DataLoader(dataset, shuffle=True, batch_size=1,
                            collate_fn=partial(collate_fn, vocab.sign2id),
                            pin_memory=True if use_cuda else False)

    start = time.time()
    for imgs, _, tgt4cal_loss in valid_loader:
        try:
            print(' load data time :',  (time.time() - start ), ' imgs size :', imgs.size())
            start = time.time()
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            print('reference :', reference)
            results = latex_producer(imgs)
            print('predict  time :',  (time.time() - start ))
            # start = time.time()
            print('results:', results)
            # print('tgt4cal_loss:', reference)
            break
        except RuntimeError:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path',default='D:\\PROJECT_TW\\git\\data\\im2latex\\ckpts\\best_ckpt.pt', type=str, help='path of the evaluated model')
    parser.add_argument('--image_file',default='5.png', type=str, help='path of the evaluated model')
    parser.add_argument('--dataset_root',default='D:\\PROJECT_TW\\git\\data\\im2latex', type=str, help='path of the evaluated model')
    parser.add_argument("--max_len", type=int, default=50, help="Max step of decoding")    
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020,help="The random seed for reproducing ")
    parser.add_argument('--data_file', default='latex_formul_normal.txt',help='data set root') 
    parser.add_argument("--image_height", type=int, default=128, help="图片高度")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    vocab = load_vocab(args.dataset_root)
    model = load_model(args.model_path)
    use_cuda = True if torch.cuda.is_available() else False

    print(model)

    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)    

    batch_valid_2(latex_producer, vocab, args)

    # batch_valid(latex_producer, vocab, args)

    # formula = valid(latex_producer, os.path.sep.join([args.dataset_root,'gen_images',args.image_file]), args.dataset_root)
    # batch_valid(latex_producer, vocab, args)
    # print('formula:', formula)



