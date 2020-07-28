import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Im2LatexModel
from training import Trainer
from utils import get_checkpoint
from latex_lmdb_dataset import lmdbDataset,collate_fn
# from latextransform import LatexImgTransform
from build_vocab import Vocab, load_vocab
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def main():

    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # parser.add_argument('--path', required=True, help='root of the model')

    # model args
    parser.add_argument("--emb_dim", type=int,
                        default=80, help="Embedding size")
    parser.add_argument("--dec_rnn_h", type=int, default=512,
                        help="The hidden state of the decoder RNN")
    # parser.add_argument("--data_path", type=str,
    #                     default="./data/", help="The dataset's dir")
    
    parser.add_argument('--dataset_root', default='D:\\PROJECT_TW\\git\\data\\im2latex',
                        help='data set root')
    parser.add_argument('--data_file', default='latex_formul_normal.txt',
                        help='data set root')

    parser.add_argument("--add_position_features", action='store_true',
                        default=True, help="Use position embeddings or not")
    # training args
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max size of formula")
    parser.add_argument("--dropout", type=float,
                        default=0., help="Dropout probility")
    parser.add_argument("--cuda", action='store_true',default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epoches", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Learning Rate")
    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential deca;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )

    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=50,
                        help="Learning Rate Decay Patience")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")
    parser.add_argument("--save_dir", type=str,
                        default="D:\\PROJECT_TW\\git\\data\\im2latex\\ckpts", help="The dir to save checkpoints")
    parser.add_argument("--print_freq", type=int, default=20,
                        help="The frequency to print message")
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")
    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")

    parser.add_argument("--image_height", type=int, default=128, help="图片高度")

    args = parser.parse_args()
    max_epoch = args.epoches
    from_check_point = args.from_check_point
    print('from_check_point ' , from_check_point)
    if from_check_point:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Building vocab
    print("Load vocab...")
    vocab = load_vocab(args.dataset_root)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    # data loader
    print("Construct data loader...")
    dataset = lmdbDataset(root=args.dataset_root, split='train', max_len=args.max_len)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_train_split=0.1
    valid_split = int(np.floor(val_train_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[valid_split:], indices[:valid_split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                            collate_fn=partial(collate_fn, vocab.sign2id),
                            pin_memory=True if use_cuda else False)
                            # sampler=train_sampler)

    valid_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                            collate_fn=partial(collate_fn, vocab.sign2id),
                            pin_memory=True if use_cuda else False)
                            # sampler=valid_sampler)

    # train_loader = DataLoader(
    #     LatexDataset(args, data_file=args.data_file,split='train', 
    #                  transform=LatexImgTransform(imgH=args.image_height, mean=MEANS,data_root=args.dataset_root),
    #                  max_len=args.max_len),
    #                  shuffle=True,
    #                  batch_size=args.batch_size,
    #                  collate_fn=partial(collate_fn, vocab.sign2id),
    #                  pin_memory=True if use_cuda else False,
    #                  num_workers=4)
    # val_loader = DataLoader(
    #     LatexDataset(args, data_file=args.data_file,split='valid', 
    #                  transform=LatexImgTransform(imgH=args.image_height, mean=MEANS,data_root=args.dataset_root),
    #                  max_len=args.max_len),
    #                  shuffle=True,
    #                  batch_size=args.batch_size,
    #                  collate_fn=partial(collate_fn, vocab.sign2id),
    #                  pin_memory=True if use_cuda else False,
    #                  num_workers=4)

    # construct model
    print("Construct model")
    vocab_size = len(vocab)
    print('vocab size:', vocab_size)
    model = Im2LatexModel(
        vocab_size, args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=args.dropout
    )
    model = model.to(device)
    print("Model Settings:")
    print(model)

    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    if from_check_point:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, valid_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch)
    else:
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, valid_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epoches)
    # begin training
    trainer.train()


if __name__ == "__main__":
    '''
        python train.py --data_path=D:\\PROJECT_TW\\git\\data\\im2latex --save_dir=D:\\PROJECT_TW\\git\\data\\im2latex\\ckpts --from_check_point=true
    '''
    main()
