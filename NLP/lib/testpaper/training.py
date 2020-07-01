# encoding: utf-8
# https://blog.csdn.net/u011534057/article/details/51452564
import os
from os.path import join
import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from config import logger

class Trainer(object):
    def __init__(self, optimizer, model,criterion,lr_scheduler,
                 train_loader, val_loader, args,
                 use_cuda=True,init_epoch=1, last_epoch=15):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.args = args
        self.criterion = criterion
        self.use_cuda = use_cuda

        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch-1)*len(train_loader)
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.best_accuracy = -1

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}"
        if self.use_cuda:
            self.model.cuda()
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # text = torch.IntTensor(self.args.batch_size * 5)
        # length = torch.IntTensor(self.args.batch_size)
        # text = Variable(text)
        # length = Variable(length)
        total_step = 0    
        batch_losses = 0.0
        while self.epoch <= self.last_epoch:
            self.model.train()

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                input_data = batch.text
                # input_data = input_data.float()
                input_data = input_data.to(device)
                preds = self.model(input_data)
                loss = self.criterion(preds, batch.label)
                clip_grad_norm_(self.model.parameters(), self.args.clip)
                loss.backward()
                self.optimizer.step()        
                batch_losses = loss.item() + batch_losses
                if total_step % 100 == 0:
                    # logger.info('train input data %s ' % input_data)
                    # logger.info('train input data label %s ' % batch.label)
                    logger.info(mes.format(self.epoch,total_step, len(self.train_loader), 
                                    (total_step/len(self.train_loader)),batch_losses/100))     
                    batch_losses = 0
                total_step += 1
            
            self.epoch += 1

            # if self.epoch % 50 == 0:
            accuracy, n_correct, valid_loss = self.valid()
            logger.info('valid accuracy {:.4f}, num_correct {:.2f}, valid {:.4f}'.format(accuracy, n_correct, valid_loss))
            self.lr_scheduler.step(valid_loss)
            if accuracy > self.best_accuracy:
                self.save_model('test_paper_best')
                self.best_accuracy = accuracy
            
            
            total_step = 0


    def valid(self):
        logger.info('begin valid data')
        self.model.eval()
        if self.use_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        device = torch.device("cuda" if self.use_cuda else "cpu")
        count = 0
        valid_loss = 0.
        n_correct = 0.
        total = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                input_data = batch.text
                input_data = input_data.to(device)
                preds = self.model(input_data)
                loss = self.criterion(preds, batch.label)
                valid_loss += loss.item()
                preds_target = torch.argmax(preds, dim=1)
                for pred, target in zip(preds_target, batch.label):
                    if pred == target:
                        n_correct = n_correct + 1
                total = total + len(batch.label)
            logger.info('input data %s' % input_data)
            logger.info('preds target %s' % preds_target)
            logger.info('real target %s' % batch.label)
        valid_loss = valid_loss / len(self.valid_loader)
        accuracy = n_correct / total

        return accuracy, n_correct, valid_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        logger.info("Saving checkpoint to {}".format(save_path))
        torch.save(self.model.state_dict(), save_path)


