import cv2
import numpy as np
from model.ctpn.lib.fast_rcnn.config import cfg


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    # if cfg.TRAIN.RANDOM_DOWNSAMPLE:
    #     r = 0.6 + np.random.rand() * 0.4
    #     im_scale *= r
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

if __name__ == "__main__":
    img = cv2.imread("D:\\PROJECT_TW\\anly\\model\\ctpn\\data\\VOCdevkit2007\\VOC2007\\JPEGImages\\img_1002.jpg")
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    MAX_SIZE = 1000
    target_size = 600
    im , im_scale = prep_im_for_blob(img,PIXEL_MEANS,target_size,MAX_SIZE)
    imgList = []
    imgList.append(im)
    im_blob = im_list_to_blob(imgList)
    blobs = {'data': im_blob}


