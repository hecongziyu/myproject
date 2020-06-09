# encoding: utf-8
import utils 
from ocr_model import CRNNClassify
from lmdb_dataset import lmdbDataset, adjustCollate
from lmdb_transform import CharImgTransform
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training import Trainer
import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(args):
    imgH = 32
    imgC = 1
    nh = 256    
    alpha = args.alpha
    converter = utils.strLabelConverter(alpha)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    train_dataset = lmdbDataset(root=os.path.sep.join([args.data_root,'lmdb']), 
                          split='train',
                          transform=CharImgTransform(data_root=args.data_root))       

    valid_dataset = lmdbDataset(root=os.path.sep.join([args.data_root,'lmdb']), 
                          split='valid',
                          transform=CharImgTransform(data_root=args.data_root))      

    train_loader = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=True if use_cuda else False,
                        collate_fn=adjustCollate(imgH=32, keep_ratio=True),
                        num_workers=args.num_workers) 

    valid_loader = DataLoader(valid_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=True if use_cuda else False,
                        collate_fn=adjustCollate(imgH=32, keep_ratio=True),
                        num_workers=args.num_workers) 

    device = torch.device("cuda" if use_cuda else "cpu")
    model = CRNNClassify(imgH=imgH,nc=imgC,nh=nh, nclass=len(alpha)+1)
    model.apply(weights_init)

    from_check_point = args.from_check_point
    if from_check_point:
        net.load_state_dict(torch.load(args.save_dir))

    model = model.to(device)

    print('model :', model)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    criterion = torch.nn.CTCLoss()


    trainer = Trainer(optimizer=optimizer, model=model,criterion=criterion,
                      lr_scheduler=lr_scheduler,converter=converter,
                      train_loader=train_loader, val_loader=valid_loader, 
                      args=args,use_cuda=use_cuda,
                      init_epoch=1, last_epoch=args.max_epoch)

    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ocr train')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr', type=str, help='path of the data')
    parser.add_argument('--alpha', default='abcdefghz', type=str)
    parser.add_argument('--split', default='train', type=str) 
    parser.add_argument('--batch_size', default=16, type=int) 
    parser.add_argument('--max_epoch', default=10, type=int) 
    parser.add_argument('--num_workers', default=0, type=int) 
    parser.add_argument('--lr', default=3e-4, type=float) 
    parser.add_argument("--decay_k", type=float, default=2.,help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential deca;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )
    parser.add_argument("--clip", type=float, default=2.0,help="The max gradient norm")    
    parser.add_argument("--save_dir", type=str, default="D:\\PROJECT_TW\\git\\data\\ocr\\weights", help="The dir to save checkpoints")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning Rate Decay Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Learning Rate")    
    parser.add_argument("--lr_patience", type=int, default=10, help="Learning Rate Decay Patience")    
    parser.add_argument("--from_check_point", action='store_true',default=False, help="Training from checkpoint or not")
    parser.add_argument("--cuda", action='store_true',default=True, help="Use cuda or not")

    args = parser.parse_args()
    train(args)

