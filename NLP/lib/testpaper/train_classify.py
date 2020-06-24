# -*- coding: UTF-8 -*-
# https://github.com/IBM/pytorch-seq2seq/tree/master/seq2seq
import argparse
from model import *
from txtdataset import TextPaperDataSet, build_vocab,TARGETS,STOP_WORDS
import torchtext.data as data
import numpy as np
import pkuseg
from preprocess import  load_custom_dict
import torch
from torchtext.data import BucketIterator
from model import TextClassify
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training import Trainer

def weights_init(m):
    pass
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)

def train(args):
    lexicon = load_custom_dict(args.data_root)
    seg = pkuseg.pkuseg(user_dict=lexicon)    
    def tokenizer(text):    
        return [wd for wd in seg.cut(text)]    

    TEXT = data.Field(tokenize=tokenizer,lower=False, batch_first=True, postprocessing=None, stop_words=STOP_WORDS)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_data = TextPaperDataSet(data_root=args.data_root, split='train', token=None, fields=[('text',TEXT),('label', LABEL)])
    valid_data = TextPaperDataSet(data_root=args.data_root, split='valid', token=None, fields=[('text',TEXT),('label', LABEL)])    
    build_vocab(TEXT, train_data, lexicon)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_iter,valid_iter = BucketIterator.splits(
        (train_data,valid_data),
        batch_sizes=(args.batch_size,args.batch_size),
        device=device,
        sort_within_batch=False,
        sort_key=lambda x:len(x.text),
        repeat=False)    


    
    

    model = TextClassify(vocab_size=len(TEXT.vocab), embed_dim=args.embed_dim, 
                            hidden_dim=args.hidden_dim, num_class=len(TARGETS))
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

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(optimizer=optimizer, model=model,criterion=criterion,
                lr_scheduler=lr_scheduler,train_loader=train_iter, val_loader=valid_iter,
                args=args,use_cuda=use_cuda,last_epoch=args.max_epoch)

    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test paper line classify')
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\testpaper', type=str, help='path of the data')
    parser.add_argument('--max_epoch',default=50000, type=int, help='path of the data')
    parser.add_argument("--cuda", action='store_true',default=True, help="Use cuda or not")
    parser.add_argument("--embed_dim", type=int,default=64, help="embed size")
    parser.add_argument("--hidden_dim", type=int,default=128, help="embed size")
    parser.add_argument("--from_check_point", action='store_true',default=False, help="Training from checkpoint or not")    
    parser.add_argument("--save_dir", type=str, default="D:\\PROJECT_TW\\git\\data\\ocr\\weights\\best_classify.pt", help="The dir to save checkpoints")    
    parser.add_argument('--lr', default=3e-4, type=float) 
    parser.add_argument("--decay_k", type=float, default=2.,help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential deca;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )
    parser.add_argument("--batch_size", type=int, default=16,help="The max gradient norm")    
    parser.add_argument("--clip", type=float, default=2.0,help="The max gradient norm")    
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning Rate Decay Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Learning Rate")    
    parser.add_argument("--lr_patience", type=int, default=10, help="Learning Rate Decay Patience")    
    args = parser.parse_args()
    train(args)
