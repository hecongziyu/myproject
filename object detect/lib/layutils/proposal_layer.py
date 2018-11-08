# -*- coding:utf-8 -*-
import numpy as np
from .generate_anchors import generate_anchors
from .anchor_target_layer import bbox_transform_inv, clip_boxes
from lib.utils.nms import nms
import torch



DEBUG = False
## Number of top scoring boxes to keep before apply NMS to RPN proposals
RPN_PRE_NMS_TOP_N = 12000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
RPN_POST_NMS_TOP_N = 1000
#__C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
RPN_MIN_SIZE = 8

RPN_NMS_THRESH = 0.7


"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """
    Parameters
    ----------
    rpn_cls_prob: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """

    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    pre_nms_topN = RPN_PRE_NMS_TOP_N       # 12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = RPN_POST_NMS_TOP_N     # 2000，做完nms之后，最多保留的box的数目
    nms_thresh = RPN_NMS_THRESH            #  nms用参数，阈值是0.7

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    # import pdb
    # pdb.set_trace()

    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
    scores = scores.contiguous().view(-1, 1)
    # import pdb
    # pdb.set_trace()
    proposals = bbox_transform_inv(anchors.data.numpy(), rpn_bbox_pred.data.numpy())
    proposals = clip_boxes(proposals, im_info[:2])

    # np.where(anchors[:,0]==206)
    # Pick the top region proposals
    scores, order = scores.view(-1).sort(descending=True)
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        scores = scores[:pre_nms_topN].view(-1, 1)
    proposals = proposals[order.data, :]
    proposals = torch.from_numpy(proposals)   
    
    # Non-maximal suppression
    if DEBUG:
        print('proposal size {} \n --> {}'.format(proposals.size(),proposals))
        print('scores size {} \n --> {}'.format(scores.size(),scores))
    keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep, ]

    # Only support single image as input
    batch_inds = proposals.new_zeros(proposals.size(0), 1)
    blob = torch.cat((batch_inds, proposals), 1)

    return blob, scores
