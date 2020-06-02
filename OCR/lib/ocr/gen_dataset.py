# -*- coding:utf-8 -*-
import os
import lmdb  
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import argparse

'''
将生成的数据保存到lmdb缓存数据库
保存的信息包括：
1、字符图片 
2、背景图片
3、原始图片
4、字符、坐标位置
'''
alpha = 'abcde'

def random_alpha(size = 100):
    alpha_list = np.random.choice(list(alpha), size, replace=True)
    return alpha_list


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)



def createCharImageCache(env, data_root, dest_dir, charImageLists):

    for key, value in charImageLists.items():
        cache = {}
        print(f'import {key} char image data ', len(value))
        print(os.path.sep.join([data_root, dest_dir, key, value[0]]))    
        for idx in tqdm(range(len(value)),ncols=150,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            image_bin =  readImageAndConvert2Gray(os.path.sep.join([data_root, dest_dir, key, value[idx]]))
            image_key = f'{key}_{idx}'
            cache[image_key] = image_bin
        cache[f'{key}_num_samples'] = str(len(value)).encode()
        writeCache(env, cache)

def createOriginImageCache(env, data_root, dest_dir, charImageLists):
    for key, value in charImageLists.items():
        cache = {}
        print(f'import {key} origin image data ', len(value))
        for idx in tqdm(range(len(value)),ncols=150,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            image_bin =  readImageAndConvert2Gray(os.path.sep.join([data_root, dest_dir, key, value[idx]]))
            image_key = f'{key}_origin_{idx}'
            cache[image_key] = image_bin
        cache[f'{key}_origin_num_samples'] = str(len(value)).encode()
        writeCache(env, cache)        

def createBgImageCache(env, data_root, dest_dir, bgImageLists):
    cache = {}
    print('import bg images, ', len(bgImageLists))
    for idx in tqdm(range(len(bgImageLists)),ncols=150,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        image_bin =  readImageAndConvert2Gray(os.path.sep.join([data_root, dest_dir, 'bg', bgImageLists[idx]]))
        image_key = f'bg_{idx}'
        cache[image_key] = image_bin
    cache[f'bg_num_samples'] = str(len(bgImageLists)).encode()
    writeCache(env, cache)


def createLabelCache(env, train_data, valid_data):
    cache = {}
    for idx, label in enumerate(train_data):
        cache[f'train_{idx}'] = label.encode()
    cache[f'train_num_samples'] = str(len(train_data)).encode()
    writeCache(env, cache)


    cache = {}
    for idx, label in enumerate(valid_data):
        cache[f'valid_{idx}'] = label.encode()
    cache[f'valid_num_samples'] = str(len(valid_data)).encode()



'''
读取图片，并转换成灰度图片
'''
def readImageAndConvert2Gray(image_file):
    image = cv2.imread(image_file,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_bin = cv2.imencode(".png",image)[1].tobytes()
    return image_bin


def createDataset(data_root,dest_dir,train_data, valid_data, charImageLists, charPosLists, bgImageLists, originImageLists):
    '''
    Create LMDB dataset for CRNN training.
    split : train, valid
    train_data: 'a,a,b,a,c,ab,ba,ca,abc' ???
    charImageLists: {'SampleA':[], 'SampleB':[]}
    '''
    outputPath = os.path.sep.join([data_root, 'lmdb'])
    env = lmdb.open(outputPath, map_size=5099511627)
    createCharImageCache(env, data_root, dest_dir, charImageLists)
    createBgImageCache(env, data_root, dest_dir, bgImageLists)
    createOriginImageCache(env, data_root, 'images', originImageLists)
    createLabelCache(env, train_data, valid_data)


def get_char_image_lists(data_root, dest_dir, split_list=None):
    if split_list is None:
        split_list = ['SampleA', 'SampleB', 'SampleC', 'SampleD', 'SampleE']
    char_image_dict = {}

    for split in split_list:
        char_image_dict[split] = os.listdir(os.path.sep.join([data_root, dest_dir, split]))
    return char_image_dict


def get_bg_images_lists(data_root, dest_dir):
    return os.listdir(os.path.sep.join([data_root, dest_dir,'bg']))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR Evaluating Program")
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr', type=str, help='path of the evaluated model')
    parser.add_argument("--image_height", type=int, default=32, help="图片高度")                
    args = parser.parse_args()
    # torch.manual_seed(2020)
    # torch.cuda.manual_seed(2020)

    # readImageAndConvert2Gray('D:\\PROJECT_TW\\git\\data\\ocr\\dest\\bg\\bg_A_1327.png')
    train_data = random_alpha(50000)
    valid_data = random_alpha(2000)

    char_image_lists = get_char_image_lists(args.data_root, 'dest')
    bg_image_lists = get_bg_images_lists(args.data_root, 'dest')
    origin_image_lists = get_char_image_lists(args.data_root,'images')
    print('origin_image_lists len :', len(origin_image_lists))
    # print(len(char_image_lists['SampleA']))

    createDataset(data_root=args.data_root, dest_dir='dest', train_data=train_data,
            valid_data=valid_data, charImageLists=char_image_lists, charPosLists=None,
            bgImageLists=bg_image_lists, originImageLists=origin_image_lists)


