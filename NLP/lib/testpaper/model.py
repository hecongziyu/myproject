# -*- coding: UTF-8 -*-
'''
https://blog.csdn.net/david0611/article/details/81090371 embed 词向量
https://blog.csdn.net/luoyexuge/article/details/83857778  句向量
https://www.cnblogs.com/webbery/p/11766623.html pytorch的Embedding使用
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html  !!!!!!
https://cloud.tencent.com/developer/article/1507559
https://blog.csdn.net/tommorrow12/article/details/80896331?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
'''
import torch.nn as nn
import torch
import torch.nn.functional as F

class TextEmbedding(nn.Module):
    '''
    对输入的文字串进行编码, LSTM编码完成后，则为该文字串的编码
    '''
    def __init__(self,  vocab_size, embed_dim, hidden_dim, bidirectional=True):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim,
                           hidden_size=hidden_dim,
                           batch_first=True,
                           bidirectional=bidirectional)

    def forward(self, input_data):
        '''
        input data: batch * word id lists 
        '''

        output = self.embedding(input_data)
        # print('embed data size:', output.size())
        output, c_t = self.lstm(output)       
        return output, c_t



class TextClassify(nn.Module):
    '''
    对文字串编码后，进行归类，分为 标题、问题、答案、解析等，训练目的，固化对文字串的编码，供后续问题分级用
    '''
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, drop_out=0.1):
        super(TextClassify, self).__init__()
        self.txt_embedding = TextEmbedding(vocab_size, embed_dim, hidden_dim)
        self.layers = nn.Sequential(nn.Linear(hidden_dim*2, 256), nn.ReLU(),nn.Linear(256, 256), nn.Linear(256, 128))
        # self.layers = nn.Sequential(nn.Linear(hidden_dim, 256), nn.Linear(256, 128))        
        self.fc = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_data):
        '''
            input data: batch *  word lists 
        '''
        # print('input size :', input_data.size())
        # print('input data:', input_data)
        output, _ = self.txt_embedding(input_data)
        output = output[:,0,:]
        output = self.layers(output)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.softmax(output)
        return output


class TextEmbeddingBagClassify(nn.Module):
    def __init__(self,vocab_size, embed_dim, num_class,drop_out=0.1):
        super().__init__()
        # mode='sum'
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.liner = nn.Linear(embed_dim, 128)
        self.fc = nn.Linear(128, num_class)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_out)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.liner.weight.data.uniform_(-initrange, initrange)
        self.liner.bias.data.zero_()        
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets=None):
        embedded = self.embedding(text, offsets)
        output = torch.tanh(self.liner(embedded))
        output = self.dropout(output)
        output =  self.fc(output)
        output = self.softmax(output)
        return output

    def predict(self, text, offsets=None):
        output = self.forward(text, offsets)
        return torch.argmax(output)






if __name__ == '__main__':
    # model = TextEmbedding(vocab_size=100, embed_dim=20, hidden_dim=30)
    # print(model)
    # data = torch.randint(0,100,(4,5))
    # print('input data size:', data.size())
    # out,(h_t, c_t) = model(data)
    # print('out put size:', out.size())
    # print('ouput ht size:', h_t.size())
    # print('ouput ct size:', c_t.size())


    # model =  TextClassify(vocab_size=100, embed_dim=20, hidden_dim=30, num_class=10)
    # print(model)
    # data = torch.randint(0,100,(4,5))

    model = TextEmbeddingBagClassify(vocab_size=40, embed_dim=20, num_class=10)
    print(model)
    data = torch.randint(0,40,(4,5))
    output = model(data)

    print(output.size())
    print(output)



    



