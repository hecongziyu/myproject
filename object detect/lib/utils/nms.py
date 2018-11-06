#coding:utf-8
import numpy as np
'''
https://blog.csdn.net/hongxingabc/article/details/78996407, 非极大值抑制NMS的python实现
目标检测中常用到NMS，在faster R-CNN中，每一个bounding box都有一个打分，NMS实现逻辑是
1，按打分最高到最低将BBox排序 ，例如：A B C D E F
2，A的分数最高，保留。从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，那么B和D可以认为是重复标记去除
3，余下C E F，重复前面两步。
????
'''
def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  # print('nms dets size {} thresh {}'.format(dets.size(), thresh))

  return py_cpu_nms(dets, thresh)

def py_cpu_nms(detst, thresh):
    """Pure Python NMS baseline."""
    if isinstance(detst,np.ndarray):
        dets = detst
    else:
        dets = detst.numpy()

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  #bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从小到大排列，argsort() 将score的值进行排序并返回其index, [::-1]按反的顺序返回值
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep