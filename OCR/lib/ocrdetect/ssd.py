import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os
import numpy as np
import time

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, args, phase, cfg, size, base, extras, head, num_classes, gpu_id):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg#(coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(args, self.cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
            if args.cuda:
                self.priors.to(gpu_id)

        # print(self.priors.size())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test' or phase == 'use':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg, 
                                    num_classes, 
                                    bkg_label=0, 
                                    top_k=200, 
                                    conf_thresh=0.01, 
                                    nms_thresh=0.45)
            #self.detect = Detect(cfg, num_classes, 0, 1000000, 0.01, 1.00)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()


        # apply vgg up to conv4_3 relu
        # print('begin ...')
        # start_time = time.time()
        for k in range(23):
            x = self.vgg[k](x)
            # print(f'vgg layer {k} size:', x.size()) 

        # print('vgg 23 use time :', (time.time() - start_time))

        # print('vgg up to conv4_3 relu, output size ：' , x.size())  # torch.Size([1, 512, 64, 64])


        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # start_time = time.time()
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # print(f'vgg layer {k} size:', x.size()) 

        # print('vgg up to fc7 use time :', (time.time() - start_time))

        # print('vgg up to fc7, output size ：' , x.size())  # torch.Size([1, 1024, 32, 32])

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            # print(f'extry layer {k} size:', x.size()) 
            # 为什么隔一层保留, 并且保存的都为 kernel_size 不为 (1,1) 的
            if k % 2 == 1:
                sources.append(x)

        # print("length source :", len(sources))

        # apply multibox head to source layers
        '''
        loc conf input: torch.Size([1, 512, 64, 64])
        loc conf input: torch.Size([1, 1024, 32, 32])
        loc conf input: torch.Size([1, 512, 16, 16])
        loc conf input: torch.Size([1, 256, 8, 8])
        loc conf input: torch.Size([1, 256, 4, 4])
        loc conf input: torch.Size([1, 256, 2, 2])
        loc conf input: torch.Size([1, 256, 1, 1])
        '''
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # print('loc conf input:', x.size())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # loc conf 0 output size : torch.Size([1, 64, 64, 28]) torch.Size([1, 64, 64, 14])
        # for idx in range(len(loc)):
        #     print(f'loc conf {idx} output size :', loc[idx].size(), conf[idx].size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc cat size:  torch.Size([1, 152908]) torch.Size([1, 76454])
        # print('loc cat size: ', loc.size(), conf.size())
        # print('phase :', self.phase)
        if self.phase == "test" or self.phase == 'use':
            with torch.no_grad():
                # print('loc size:', loc.size())
                # print('conf size:', conf.size())
                # print('prios size:', self.priors.size())

                output, boxes, scores = self.detect(
                    loc.view(loc.size(0), -1, 4),                   # loc preds
                    self.softmax(conf.view(conf.size(0), -1,
                                 self.num_classes)),                # conf preds
                    self.priors.type(type(x.data))                  # default boxes
                )
            return output, boxes.detach(), scores.detach()
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            return output



    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False

    # [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
    extras = cfg['extras'][str(size)]

    for k, v in enumerate(extras):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, extras[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))

    # if size == 100:
    #     layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))

    return layers


# cfg : [7,7,7,7,7,7,7] 
def multibox(args, vgg, extra_layers, cfg, size, num_classes):
    loc_layers = []
    conf_layers = []

    vgg_source = [21, -2]
    # 第21层和倒数第二层
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4,
                                 kernel_size=args.kernel, padding=args.padding)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes,
                                  kernel_size=args.kernel, padding=args.padding)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4,
                                 kernel_size=args.kernel, padding=args.padding)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes,
                                  kernel_size=args.kernel, padding=args.padding)]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(args, phase, cfg, gpu_id, size=300, num_classes=21):
    if phase != "test" and phase != "train" and phase != 'use':
        print("ERROR: Phase: " + phase + " not recognized")
        return

    # print('base :', base)
    # print(cfg['channel'])
    base_, extras_, head_ = multibox(args,vgg(base, cfg['channel'], False),
                                     add_extras(cfg, size, 1024),
                                     cfg['mbox'][str(size)], size, num_classes)

    return SSD(args, phase, cfg, size, base_, extras_, head_, num_classes, gpu_id)
