import torch
import numpy as np

DEBUG = False
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
            (boxes[:, 3] - boxes[:, 1] + 1)
    
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
            (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    
    if DEBUG:
        print('box areas {} --> \n {}'.format(box_areas.shape,box_areas))
        print('query areas {} --> \n {}'.format(query_areas.shape, query_areas))
        print('boxes 2:3  --> \n {}'.format(boxes[:, 2:3]))
        print('query boxes 2 3 --> \n {}'.format(query_boxes[:, 2:3].t()))
    
    # clamp https://blog.csdn.net/u013230189/article/details/82627375  将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量
    
    # MIN(X2) - MAX(X1)  取出两个边框之间重复的X轴部分的宽
    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    
    # MIN(Y2) - MAX(Y1)  取出两个边框之间重复的Y轴部分的高
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    
    
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    if DEBUG:
        print('iw {} --> \n {}'.format(iw.shape,iw))
        print('ih {} --> \n {}'.format(ih.shape, ih))
        print('ua {} --> \n {}'.format(ua.shape, ua))

    return out_fn(overlaps)


if __name__ == '__main__':
    
    #array([[ 0.,  0., 15., 15.],
    #   [16.,  2., 31., 13.],
    #   [16.,  0., 31., 15.],
    #   [32.,  2., 47., 13.],
    #   [32.,  0., 47., 15.],
    #   [48.,  2., 63., 13.],
    #   [48.,  0., 63., 15.],
    #   [64.,  2., 79., 13.],
    #   [64.,  0., 79., 15.]], dtype=float32)    

    # array([[889., 249., 904., 426.,   1.],
    #        [873., 249., 888., 426.,   1.],
    #        [857., 249., 872., 426.,   1.],
    #        [841., 249., 856., 426.,   1.],
    #        [825., 249., 840., 426.,   1.]], dtype=float32)   
    pass
     