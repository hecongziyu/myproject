# Sample command
# python3 train.py --dataset GTDB --dataset_root /home/psm2208/data/GTDB/
# --cuda True --visdom True --batch_size 16 --num_workers 8 --layers_to_freeze 0
# --exp_name weights_1 --model_type 512 --suffix _512 --type processed_train_512
# --cfg math_gtdb_512 --loss_fun fl --kernel 1 5 --padding 0 2 --neg_mining False

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from init import init_args
import argparse
from utils import helpers
import logging
import time
import datetime
from torchviz import make_dot
from torch.optim.lr_scheduler import ReduceLROnPlateau

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()	

def train(args):

    cfg = exp_cfg[args.cfg]
    dataset = GTDBDetection(args, args.training_data, split='train',
                            transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    valid_dataset = GTDBDetection(args, args.validation_data, split='validate',
    	                          transform=SSDAugmentation(cfg['min_dim'], mean=MEANS))

    valid_data_loader = data.DataLoader(valid_dataset, 1,
                                        num_workers=args.num_workers,
                                        shuffle=False, collate_fn=detection_collate,
                                        pin_memory=True)    
    logging.debug('Training set size is ' + str(len(dataset)))

    gpu_id = 0

    if args.cuda:
        gpu_id = helpers.get_freer_gpu()
        logging.debug('Using GPU with id ' + str(gpu_id))
        torch.cuda.set_device(gpu_id)

    ssd_net = build_ssd(args, 'train', cfg, gpu_id, cfg['min_dim'], cfg['num_classes'])
    # print(ssd_net)
    ct = 0
    # freeze first few layers
    for child in ssd_net.vgg.children():
        if ct >= args.layers_to_freeze:
            break

        child.requires_grad = False
        ct += 1

    if args.resume:
        logging.debug('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_state_dict(torch.load(args.resume))
    else:
        vgg_weights = torch.load(os.path.join(args.root_path, 'weights',args.basenet))
        logging.debug('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)        

    if args.cuda:
        ssd_net = ssd_net.cuda()

    step_index = 0

    if not args.resume:
        logging.debug('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)


    optimizer = optim.Adam(ssd_net.parameters(), lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)    

    criterion = MultiBoxLoss(args, cfg, args.pos_thresh, 0, 3)
    loc_loss = 0
    conf_loss = 0    
    epoch = 0
    epoch_size = len(dataset) // args.batch_size
    best_val_loss = 10

    print('epoch_size:', epoch_size)

    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        # resume training
        ssd_net.train()
        t0 = time.time()
        t1 = time.time()

        # load train data
        try:
            images, targets, _ = next(batch_iterator)
        except StopIteration:
             batch_iterator = iter(data_loader)
             images, targets, _ = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(images)
            targets = [ann for ann in targets]

        # print("image , targets size : ", images.size(), targets[0])
        out = ssd_net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = args.alpha * loss_l + loss_c #TODO. For now alpha should be 1. While plotting alpha is assumed to be 1
        loss.backward()
        optimizer.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # Log progress
        if iteration % 10 == 0:
            logging.debug('timer: %.4f sec.' % (t1 - t0))
            logging.debug('iter ' + repr(iteration) + 
                          ' || Loss: %.4f || loc loss: %.4f || conf loss: %.4f || base valid loss %.4f ||'
                          % (loss.item(),loss_l.item(),loss_c.item(),best_val_loss))

        if iteration!=0 and (iteration % epoch_size == 0):
            epoch += 1            
            # reset epoch loss counters
            validation_loss = validate(args, ssd_net, criterion, cfg, valid_data_loader)
            lr_scheduler.step(validation_loss)
            logging.debug('iter ' + repr(iteration) + ' ||current valid loss: %.4f || base valid loss %.4f ||' % (validation_loss.item(), best_val_loss))
            if validation_loss < best_val_loss:
            	best_val_loss = validation_loss
            	save_model(args, cfg,ssd_net, 'best_ssd')
            # loc_loss = 0
            # conf_loss = 0            


def save_model(args, cfg, net, filename):
		torch.save(net.state_dict(),
					os.path.join(args.root_path,'ckpts',
                                   'weights_' + args.exp_name, 'best_ssd' + str(args.model_type)  + '.pth'))	

def validate(args, net, criterion, cfg, valid_data_loader):
    try:
        # Turn off learning. Go to testing phase
        net.eval()


        total = len(dataset)
        done = 0
        loc_loss = 0
        conf_loss = 0

        start = time.time()
        with torch.no_grad():
	        for batch_idx, (images, targets, ids) in enumerate(valid_data_loader):

	            done = done + len(images)
	            # logging.debug('processing {}/{}'.format(done, total))

	            if args.cuda:
	                images = images.cuda()
	                targets = [ann.cuda() for ann in targets]
	            else:
	                images = Variable(images)
	                targets = [Variable(ann, volatile=True) for ann in targets]

	            y = net(images)  # forward pass

	            loss_l, loss_c = criterion(y, targets)
	            loc_loss += loss_l.item()  # data[0]
	            conf_loss += loss_c.item()  # data[0]

        end = time.time()
        logging.debug('Time taken for validation ' + str(datetime.timedelta(seconds=end - start)))

        return (loc_loss + conf_loss) / (total/validation_batch_size)
    except Exception as e:
        logging.error("Could not validate", exc_info=True)
        return 0

'''
# Sample command
# python train_new.py --dataset GTDB --dataset_root D:\PROJECT_TW\git\data\mathdetect\data
# --cuda True --visdom True --batch_size 16 --num_workers 8 --layers_to_freeze 0
# --exp_name weights_1 --model_type 512 --suffix _512 --type processed_train_512
# --cfg math_gtdb_512 --loss_fun fl --kernel 1 5 --padding 0 2 --neg_mining False
'''
if __name__ == '__main__':

    args = init_args()
    start = time.time()
    try:
        filepath=os.path.join(args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log")
        print('Logging to ' + filepath)
        # logging.basicConfig(filename=filepath,
        #                     filemode='w', format='%(process)d - %(asctime)s - %(message)s',
        #                     datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
        logging.basicConfig(format='%(process)d - %(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)        

        train(args)
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

    end = time.time()
    logging.debug('Total time taken ' + str(datetime.timedelta(seconds=end - start)))
    logging.debug("Training done!")
