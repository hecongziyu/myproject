# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import torch
from torch import nn
from torch.autograd import Variable

from config import opt


class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers,
                 dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        print('hidden size {} num layers {} drop out {}'.format(hidden_size, num_layers, dropout))
        # self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.project = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hs=None):
        batch = x.shape[0]
        # if hs is None:
        #     hs = Variable(
        #         torch.zeros(self.num_layers, batch, self.hidden_size))
        #     if opt.use_gpu:
        #         hs = hs.cuda()

        word_embed = self.word_to_vec(x)  # (batch, len, embed)
        # print('word embed --> {} size {}'.format(word_embed, word_embed.size()))
        # word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.rnn(word_embed)  # (len, batch, hidden)
        le, mb, hd = out.shape
        out = out.contiguous().view(le * mb, hd)
        out = self.project(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0
