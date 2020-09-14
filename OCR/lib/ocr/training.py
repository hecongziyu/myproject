# encoding: utf-8
import os
from os.path import join
import utils
import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import logging

class Trainer(object):
    def __init__(self, optimizer, model,criterion,lr_scheduler,
                 converter, train_loader, val_loader, args,
                 use_cuda=True,init_epoch=1, last_epoch=15):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.criterion = criterion
        self.converter = converter
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

        # text = torch.IntTensor(self.args.batch_size * 5)
        # length = torch.IntTensor(self.args.batch_size)
        # text = Variable(text)
        # length = Variable(length)
        total_step = 0    
        while self.epoch <= self.last_epoch:
            self.model.train()
            batch_losses = 0.0
            batch_iterator = iter(self.train_loader)

            for idx in range(len(batch_iterator)):
                try:
                    images, labels = next(batch_iterator)
                    # with torch.no_grad():
                    images = Variable(images.type(dtype))
                    text, length = self.converter.encode(labels)

                        # utils.loadData(text, t)
                        # utils.loadData(length, l)
                    preds = self.model(images).cpu()
                    preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
                    loss = self.criterion(preds, text, preds_size, length) / preds.size(1)
                    batch_losses = loss.item() + batch_losses
                    self.optimizer.zero_grad()
                    clip_grad_norm_(self.model.parameters(), self.args.clip)
                    loss.backward()
                    self.optimizer.step()        


                    if total_step % 2 == 0:
                        logging.info(mes.format(self.epoch,total_step, len(self.train_loader), (total_step/len(self.train_loader)),batch_losses/100))     
                        batch_losses = 0
                    total_step += 1
                except Exception as e:
                    print('train error:', e)
                

            accuracy, num_correct, valid_loss = self.valid()
            logging.info('valid accuracy: {}, num_correct: {}, valid: {} best accuracy: {}'.format(accuracy, num_correct, valid_loss, self.best_accuracy))
            self.lr_scheduler.step(valid_loss)

            if accuracy > self.best_accuracy:
                self.save_model('ocr_best')
                self.best_accuracy = accuracy
            self.epoch += 1
            total_step = 0


    def valid(self):
        logging.info('begin valid data ...')
        if self.use_cuda:
            self.model.cuda()
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        self.model.eval()
        count = 0
        valid_loss = 0.
        n_correct = 0.
        with torch.no_grad():
            batch_iterator = iter(self.val_loader)
            for idx in range(len(self.val_loader)):
                try:
                    images, labels = next(batch_iterator)
                    images = Variable(images.type(dtype))
                    count += images.size(0)  
                    text, length =  self.converter.encode(labels)
                    logging.info('labels: %s text: %s, lenght : %s' % (labels, text, length))
                    preds = self.model(images).cpu()
                    preds_size = Variable(torch.IntTensor([preds.size(0)] * preds.size(1)))
                    loss = self.criterion(preds, text, preds_size, length) / preds.size(1)
                    valid_loss += loss
                    _, preds = preds.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    logging.info('preds : %s' % preds)
                    sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)

                    for pred, target in zip(sim_preds, labels):
                        if pred == target.lower():
                            n_correct += 1  
                        
                except Exception as e:
                    print('valid exception :', e)
        valid_loss = valid_loss / len(self.val_loader)
        logging.info('sim preds -->{}'.format(sim_preds))
        logging.info('text -->{}'.format(labels))
        accuracy = n_correct / count
        return accuracy, n_correct, valid_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        logging.info("Saving checkpoint to {}".format(save_path))
        torch.save(self.model.state_dict(), save_path)


