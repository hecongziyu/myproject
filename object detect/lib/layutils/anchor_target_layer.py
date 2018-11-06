# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from ..utils.bbox import bbox_overlaps
import torch

DEBUG = True
RPN_CLOBBER_POSITIVES = False
RPN_POSITIVE_OVERLAP = 0.8
RPN_NEGATIVE_OVERLAP = 0.5
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256
RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
RPN_POSITIVE_WEIGHT = -1.0

def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. \
            format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets

def bbox_transform_inv(boxes, deltas):

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """



    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)


    A = num_anchors
    total_anchors = all_anchors.shape[0]    #  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False) H*W， 4
    K = total_anchors / num_anchors
    
    

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)


    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]
    
    if DEBUG:
        print('total_anchors', total_anchors)
        print('inds_inside', len(inds_inside))    

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    
    if DEBUG:
        print('anchors.shape', anchors.shape)    

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    
    if DEBUG:
        print('anchors , gt boxes --> overlaps:', overlaps)

    # axis=1 按行值    axis=0 按列值.  argmax(axis=1) 按行值找到对应列值最大的项
    argmax_overlaps = overlaps.argmax(axis=1) # (A)#找到和每一个gtbox，overlap最大的那个anchor

    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   # argmax_overlaps 是选择overlaps最大的一列的列位置
    
    # argmax(axis=1) 按列扫描找到每列最大的行索引值
    gt_argmax_overlaps = overlaps.argmax(axis=0)    # 取得gt_boxes 对应重合度最大的 anchors的行值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]  
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]   # 返回overlaps=gt max overlaps
    
    if DEBUG:
        print('arg max over laps ->', argmax_overlaps)
        print('max over laps ->', max_overlaps)
        print('gt argmax overlaps ->',len(gt_argmax_overlaps))

    if not RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    # gt_artmax_overlaps是gt_boxs的每个位置在anchros里面有最大值的位置，即每个gt_boxes在anchors里重合度最大的地方
    labels[gt_argmax_overlaps] = 1    

    # fg label: above threshold IOU
    labels[max_overlaps >= RPN_POSITIVE_OVERLAP] = 1

    if RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        
    if DEBUG:
        print('Number FG -> {}  Number BG -> {}'.format(num_fg, num_bg))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    
    # argmax_overlaps是对应anchors每一行对应最大的列索引值，相对于gt_boxes则是其对应的行号
    cgt_boxes = gt_boxes[argmax_overlaps, :]
    if DEBUG:
        print('anchors size -> {} cgt boxes size -> {}'.format(anchors.shape, cgt_boxes.shape))
    bbox_targets = _compute_targets(anchors, cgt_boxes )

    # zz = bbox_transform_inv(torch.from_numpy(anchors[4506]).view(1,4),torch.from_numpy(bbox_targets[4506]).view(1,4))

    # bbox inside 权重
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(RPN_BBOX_INSIDE_WEIGHTS)
    # bbox outside 权重
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

    if RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        # 外部权重，前景是1，背景是0
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))        
    else:
        assert ((RPN_POSITIVE_WEIGHT > 0) &
                (RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))

    # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    # labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    # labels = labels.reshape((1, 1, A * height, width))
    labels = labels.reshape((1, height, width, A))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(torch.from_numpy(ex_rois), torch.from_numpy(gt_rois[:, :4])).numpy()



if __name__ == '__main__':
    # source im size (600, 767, 3), torch.Size([1, 2, 370, 47]
    # from model.ctpn.lib.layer_utils.snippets import generate_anchors_pre
    from model.ctpn.test import get_test_blob
    import numpy as np
    import torch
    imdb, roidb, blobs = get_test_blob()
    gt_boxes = roidb[0]['boxes']
    _feat_stride = [16]
    num_anchors = 10
    # all_anchors,_ = generate_anchors_pre(height=37, width=47, feat_stride=(16.), anchor_scales=(16.), anchor_ratios=(0.5,1,2))
    # print(all_anchors[5775])
    all_anchors = None
    rpn_cls_score = torch.rand(1,37,47,20)
    im_info = [600, 767,1]
    boxes = np.empty((gt_boxes.shape[0],5),dtype=np.float32)
    boxes[:,0:4] = gt_boxes
    boxes[:,4] =1
    for i in range(1):
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(rpn_cls_score, boxes, im_info, _feat_stride, all_anchors, num_anchors)
        rpn_labels = torch.from_numpy(rpn_labels).float()  # .set_shape([1, 1, None, None])
        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float()  # .set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_inside_weights = torch.from_numpy(
            rpn_bbox_inside_weights).float()  # .set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_outside_weights = torch.from_numpy(
            rpn_bbox_outside_weights).float()  # .set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = rpn_labels.view(-1)
        rpn_bbox_targets = rpn_bbox_targets.view(-1,4)
        # print(all_anchors[5775])
        print(rpn_labels[5775])
        print(rpn_bbox_targets[5775])
