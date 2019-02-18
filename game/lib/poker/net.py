import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import torch.optim as optim
from torchvision import datasets, transforms
import os
from easydict import EasyDict as edict

np.random.seed(1)
torch.manual_seed(1)

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

N_ACTIONS = 14
N_STATES = 8

ENV_A_SHAPE = 0

args = edict({
    "lr": 0.001,
    "dropout": 0.3,
    "epochs": 2,
    "batch_size": 64,
    "cuda": torch.cuda.is_available(),
    "num_channels": 512
})

class Net(nn.Module):
    def __init__(self,state_num, action_num):
        super(Net, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.fc1 = nn.Linear(self.state_num, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out1 = nn.Linear(100, self.action_num)
        self.out1.weight.data.normal_(0, 0.1)   # initialization
        self.out2 = nn.Linear(100,1)             # 价值
        self.out2.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out1(x)
        values = self.out2(x)
        return F.log_softmax(actions_value,dim=1), F.tanh(values)



class NNetWrapper(object):
    def __init__(self,state_num, action_num,game):
        self.game = game
        self.state_num = state_num
        self.action_num = action_num
        self.nnet = Net(self.state_num, self.action_num)

    def predict(self,state):
        x = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        self.nnet.eval()
        with torch.no_grad():
            p,v = self.nnet.forward(x)
        # log softmax 为负，采用torch.exp 转为正数
        return torch.exp(p).data.cpu().numpy()[0],v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
        else:
            map_location = 'cpu'
            checkpoint = torch.load(filepath, map_location=map_location)

            self.nnet.load_state_dict(checkpoint['state_dict'])

    def train(self, ex):
        optimizer = optim.Adam(self.nnet.parameters())
        examples = ex.copy()
        print('train examples -->{}'.format(examples[0]))
        examples = [(self.game.getTableStates(x[0],x[3]),x[1],x[2],x[3])  for x in examples]
        for epoch in range(args.epochs):
            print('train epoch {} len {}'.format(epoch, len(examples)))
            self.nnet.train()
            batch_idx = 0
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                # zip(*[]) 将数组按列分隔, 依次是状态、action概率、价值
                states,  vs, pis,ac = list(zip(*[examples[i] for i in sample_ids]))
                batch_idx += 1
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # # predict
                out_pi, out_v = self.nnet(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]