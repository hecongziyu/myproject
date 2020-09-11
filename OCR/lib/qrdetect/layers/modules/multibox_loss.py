# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp
from .focal_loss import FocalLoss

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, args, cfg, overlap_thresh, bkg_label, neg_pos):

        super(MultiBoxLoss, self).__init__()
        self.args = args
        self.num_classes = cfg['num_classes']
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.negpos_ratio = neg_pos
        self.variance = cfg['variance']
        self.focal_loss = FocalLoss()
        # self.neg_overlap = neg_overlap
        # self.encode_target = encode_target
        # self.use_prior_for_matching = prior_for_matching
        # self.do_neg_mining = args.neg_mining

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """



        loc_data, conf_data, priors = predictions
        print('loc data size:', loc_data.size())
        # print('conf data size:', conf_data.size())
        # print('prios data size:', priors.size())
        # print('num class:', self.num_classes)
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        # print('loc t size:', loc_t.size())
        # print('conf t size:', conf_t.size())

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data

            # print('labels:', labels)
            # print('conf_data:', conf_data)

            defaults = priors.data
            # match 处理后 #loc_t： [num_priors,4] encoded offsets to learn 
            # conf t 是在原来labels值上面加了1， 0 作为背景
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)


        if self.args.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # 取非背景类别的数据
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        #print('conf_t view ', conf_t.view(-1, 1))
        #print('conf_t ' + conf_t.view(-1, 1))
        '''
        log_sum_exp:  Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.   ？？？？？？'''
        
        # print('batch conf size :', batch_conf.size(), ' conf t size: ', conf_t.view(-1,1).size())

        #  batch_conf gather 是以conf_t 作为 index，取该index的数值, 如 conf 类别 为 1,1,2，则会
        #  取bath_conf对应1,1,2列的值
        #  log_sum_exp 为 该类别里面最大值
        #  loss c 的算法 
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        if self.args.neg_mining:
            loss_c = loss_c.view(pos.size()[0], pos.size()[1])
            loss_c = loss_c.view(num, -1)
            loss_c[pos] = 0  # filter out pos boxes for now
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)

            num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
        else:
            #num_neg = torch.tensor(0).expand_as(idx_rank)
            #num_neg[idx_rank] = 1
            neg = conf_t == 0

        # Confidence Loss Including Positive and Negative Example
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        if self.args.loss_fun == 'ce':
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        else:
            loss_c = self.focal_loss.compute(conf_p, targets_weighted)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum() + 1

        # print('NUM POS DATA SUM:', N)


        #loss_l = loss_l.double()
        #loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
