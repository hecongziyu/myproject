import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

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

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value



class DQNet(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        # memory = [MEMORY_CAPACITY, s+a+r+s_]
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()




    def choss_action(self, state):
        state = Variable(torch.unsqueeze(torch.FloatTensor(state), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(state)
#             print('choos action actions value --> {}'.format(actions_value))
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         print('observation --> {} action --> {}'.format(x, action))
        return action

    def save_net(self):
        pass

    

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
#         print('transition --> {}'.format(transition))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        print('begin learning ... ')
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience, gathcer 只取列是action的state的值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # 取下一次的最大可能action，与target_net计算V值分开
        q_next_actions = self.eval_net(b_s_).argmax(dim=1)
#         print('q_eval value --> {}'.format(q_eval))
        # 代码中的detach和required_grad的引入是减少了计算量，required_grad=false会计算误差，不计算wb的梯度
        q_next = self.target_net(b_s_).detach()    # detach from graph, don't backpropagate
        q_next = q_next.gather(1, q_next_actions.view(BATCH_SIZE,1))
        q_target = b_r + GAMMA * q_next   # shape (batch, 1)
        
        
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('loss --> {}'.format(loss))
        return loss




