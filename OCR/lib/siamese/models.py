# https://github.com/jingpingjiao/siamese_cnn/blob/master/siamese.py
# https://www.cnblogs.com/inchbyinch/p/12116339.html
# ReflectionPad2d 是paddingLayer，padding的方式多种，可以是指定一个值，也可以是不规则方式，即给出一个四元组，
# ReflectionPad2d 与 padding 区别， https://blog.csdn.net/Arthur_Holmes/article/details/104264944?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param 

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


#自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # print('euclidean distance:',euclidean_distance )
        # print('euclidean label:',label )
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SiameseNetwork(nn.Module):

    #  input image shape 400, 400, 1
    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()

        self.contra_loss = contra_loss

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            #  output 100*100*64

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            #  output 50 * 50 * 128

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            #  output 12 * 12 * 256


            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            # nn.MaxPool2d(2, 2),
            #  output 6 * 6 * 256

            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024), 
            nn.MaxPool2d(2, 2),

            Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        print('x :', x.size())
        output = self.cnn(x)
        return output

    def embed_image(self,x):
        with torch.no_grad():
            output = self.cnn(x)
            return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        # print('output1:', output1.size())
        output2 = self.forward_once(input2)
        if self.contra_loss:
            return output1, output2
        else:
            output = torch.cat((output1, output2), 1)
            output = self.fc(output)
            return output

