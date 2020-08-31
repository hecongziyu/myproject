import os
import sys
import torch
from torch.autograd import Variable
from sia_dataset import LmdbDataset,collate_fn
from sia_transform import LmdbTransform
from models import ContrastiveLoss, SiameseNetwork
import logging as logger
import torchvision
from torch.utils.data import DataLoader
from os.path import join


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

def train(args):
    dataset = LmdbDataset(data_dir=args.data_root, split='train', transform=LmdbTransform())
    r_sample = torch.utils.data.RandomSampler(data_source=list(range(len(dataset))), replacement=True)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                            sampler=r_sample,
                            collate_fn=collate_fn,
                            pin_memory=False)   
                            
    gpu_id = 0

    siamese_net = SiameseNetwork(contra_loss=True)
    if args.resume:
        print('resume :' , args.resume)
        siamese_net.load_state_dict(torch.load(join(args.data_root,'weights', 'siame_best_net.pth'),map_location=torch.device('cpu')))

    use_cuda = True if  torch.cuda.is_available() else False
    if use_cuda:
        gpu_id = helpers.get_freer_gpu()
        torch.cuda.set_device(gpu_id)


    if use_cuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(siamese_net.parameters())
    criterion = ContrastiveLoss(margin=args.margin)
    cur_loss = 1000

    print('begin traing ....')

    for epoch in range(args.epoch):
        siamese_net.train()
        total_loss = 0.
        for img0, img1, labels in data_loader:
            if use_cuda:
                img0, img1, labels = img0.cuda(), img1.cuda(),labels.cuda()
            optimizer.zero_grad()
            output1,output2 = siamese_net(img0, img1)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss = round(total_loss / (len(dataset) / args.batch_size) , 3)
        print('epoch : %s loss : %s, cur loss: %s' % (epoch, total_loss, cur_loss) )
        if total_loss < cur_loss:
            cur_loss = total_loss
            print('save model ...')
            torch.save(siamese_net.state_dict(), join(args.data_root,'weights', 'siame_best_net.pth'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\siamese',help='data set root')
    parser.add_argument('--batch_size', default=8, type=int,help='data set root')
    parser.add_argument('--epoch', default=20000, type=int)
    parser.add_argument('--margin', default=2.0, type=float)
    parser.add_argument('--resume', action='store_true',
                        default=False, help="Training from checkpoint or not")
    args = parser.parse_args()    

    train(args)
