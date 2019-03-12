# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
from copy import deepcopy

import numpy as np
import torch
import os
# from mxtorch import meter
# from mxtorch.trainer import Trainer, ScheduledOptim
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch import distributed
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import TextDataset, TextConverter

# https://www.imooc.com/article/32412
# https://github.com/pytorch/ignite
class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)

class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count


def get_data(convert):
    dataset = TextDataset(opt.txt, opt.len, convert.text_to_arr)
    sampler = DistributedSampler(dataset)    
    return DataLoader(dataset, 
        opt.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler)


def get_model(convert):
    # model = getattr(models, opt.model)(convert.vocab_size,
    #                                    opt.embed_dim,
    #                                    opt.hidden_size,
    #                                    opt.num_layers,
    #                                    opt.dropout)
    print('----> get model')
    model = models.CharRNN(convert.vocab_size,
                                       opt.embed_dim,
                                       opt.hidden_size,
                                       opt.num_layers,
                                       opt.dropout)
  
    if opt.use_gpu:
        model = model.cuda()
    print(model)
    return model


def get_loss(score, label):
    return nn.CrossEntropyLoss()(score, label.view(-1))


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return optimizer
    # return ScheduledOptim(optimizer)


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


class CharRNNTrainer():
    def __init__(self, convert):
        self.convert = convert
        self.n_iter = 0
        self.model = get_model(convert)
        self.criterion = get_loss
        self.optimizer = get_optimizer(self.model)
        # super().__init__(model, criterion, optimizer)
        self.config = ('text: ' + opt.txt + '\n' + 'train text length: ' + str(opt.len) + '\n')
        self.config += ('predict text length: ' + str(opt.predict_len) + '\n')

        # self.metric_meter['loss'] = meter.AverageValueMeter()

    def train(self, kwargs):
        # self.reset_meter()
        current_loss = 10
        self.model.train()
        train_data = kwargs['train_data']

        # for data in tqdm(train_data):
        for data in train_data:
            x, y = data
            x = x.long()
            y = y.long()
            # print('input x --> {}'.format(x))
            # print('input y --> {}'.format(y))
            x, y = Variable(x), Variable(y)
            score, _ = self.model(x)
            # print('score --> {} size {}'.format(score, score.size()))
            
            loss = self.criterion(score, y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            # average the gradients
            self.average_gradients()

            self.optimizer.step()

            print('loss --> {}'.format(loss))
            if loss < current_loss and opt.rank==0:
                current_loss = loss
                self.save_state_dict(opt.load_model)

            if (self.n_iter + 1) % opt.plot_freq == 0:
                print('loss-->{}'.format(loss));
                # self.n_plot += 1
            self.n_iter += 1


    def average_gradients(self):
        world_size = distributed.get_world_size()

        for p in self.model.parameters():
            distributed.all_reduce(p.grad.data, op=distributed.reduce_op.SUM)
            p.grad.data /= float(world_size)


        # Log the train metrics to dict.
        # self.metric_log['perplexity'] = np.exp(self.metric_meter['loss'].value()[0])

    def test(self, kwargs):
        """Set beginning words and predicted length, using model to generate texts.

        Returns:
            predicted generating text
        """
        self.model.eval()
        begin = np.array([i for i in kwargs['begin']])
        begin = np.random.choice(begin, size=1)
        text_len = kwargs['predict_len']
        samples = [self.convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        if opt.use_gpu:
            input_txt = input_txt.cuda()
        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(pred))[None]
            if opt.use_gpu:
                model_input = model_input.cuda()
            result.append(pred[0])

        # Update generating txt to tensorboard.
        self.writer.add_text('text', self.convert.arr_to_text(result), self.n_plot)
        self.n_plot += 1
        print(self.convert.arr_to_text(result))

    def predict(self, begin, predict_len):
        self.model.eval()
        samples = [self.convert.word_to_int(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        if opt.use_gpu:
            input_txt = input_txt.cuda()
        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(predict_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(pred))[None]
            if opt.use_gpu:
                model_input = model_input.cuda()
            result.append(pred[0])
        text = self.convert.arr_to_text(result)
        print('Generate text is: {}'.format(text))
        with open(opt.write_file, 'a') as f:
            f.write(text)

    def load_state_dict(self, checkpoints):
        if os.path.exists(checkpoints):
            print('load last model {}'.format(checkpoints))
            self.model.load_state_dict(torch.load(checkpoints))

    def save_state_dict(self, checkpoints):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        print('save model  {}', checkpoints)
        torch.save(self.model.state_dict(),checkpoints)       

    def get_best_model(self):
        if self.metric_log['perplexity'] < self.best_metric:
            self.best_model = deepcopy(self.model.state_dict())
            self.best_metric = self.metric_log['perplexity']



def train(**kwargs):
    print('kwargs -->{}'.format(kwargs))
    opt._parse(kwargs)
    print(opt.rank)
    init_process(opt)
    # torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    train_data = get_data(convert)
    char_rnn_trainer = CharRNNTrainer(convert)
    # print(train_data)
    # for data in train_data:
    #     x,y = data
    char_rnn_trainer.load_state_dict(opt.load_model)
    for _ in range(100000):
        train_data = get_data(convert)
        kwarg = {"train_data":train_data, "epchos":opt.max_epoch, "begin":opt.begin,"predict_len":opt.predict_len}
        char_rnn_trainer.train(kwarg)
    # char_rnn_trainer.fit(train_data=train_data,
    #                      epochs=opt.max_epoch,
    #                      begin=opt.begin,
    #                      predict_len=opt.predict_len)


def predict(**kwargs):
    opt._parse(kwargs)
    # torch.cuda.set_device(opt.ctx)
    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    char_rnn_trainer = CharRNNTrainer(convert)
    char_rnn_trainer.load_state_dict(opt.load_model)
    char_rnn_trainer.predict(opt.begin, opt.predict_len)

def init_process(args):
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size)


if __name__ == '__main__':
    import fire

    fire.Fire()
