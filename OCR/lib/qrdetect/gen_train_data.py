# 增加训练数据
import os
from os.path import join
import lmdb
import cv2
import numpy as np


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def gen_train_data(data_dir):
    env = lmdb.open(join(data_dir,'lmdb'), map_size=400511627)  
    qr_nSamples = 0
    bg_nSamples = 0
    # try: 
    #     with env.begin(write=False) as txn:
    #         qr_nSamples = int(txn.get('qr_total'.encode()))
    # except Exception as e:
    #     pass

    # try: 
    #     with env.begin(write=False) as txn:
    #         bg_nSamples = int(txn.get('bg_total'.encode()))
    # except Exception as e:
    #     pass        


    q_key_idx = qr_nSamples
    file_lists = os.listdir(join(data_dir,'images', 'qr'))


    for idx, fitem in enumerate(file_lists):
        cache = {}
        file_name = fitem.split('.')[0]
        print('file name:', file_name)
        with open(join(data_dir, 'images','qr', fitem),'rb') as f:
            image_data = f.read()
        cache[f'qr_{q_key_idx}'] = image_data
        cache['qr_total'] = str(q_key_idx).encode()
        writeCache(env, cache)
        q_key_idx = q_key_idx + 1


    b_key_idx = bg_nSamples
    file_lists = os.listdir(join(data_dir,'images', 'bg'))


    for idx, fitem in enumerate(file_lists):
        cache = {}
        file_name = fitem.split('.')[0]
        # print('file name:', file_name)
        # with open(join(data_dir, 'images','bg',fitem),'rb') as f:
        #     image_data = f.read()
        print('file :', join(data_dir,'images','bg',fitem))
        image = cv2.imread(join(data_dir,'images','bg',fitem), cv2.IMREAD_COLOR)
        heigh, width, _ = image.shape
        radio = 2048/heigh
        image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
        _, image_data = cv2.imencode(".png", image)

        cache[f'bg_{b_key_idx}'] = image_data
        cache[f'target_{b_key_idx}'] = np.array([[-1,-1,-1,-1,-1]],dtype=np.float).tobytes(order='C')
        cache['bg_total'] = str(b_key_idx).encode()
        writeCache(env, cache)
        b_key_idx = b_key_idx + 1  





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_dir', default='D:\\PROJECT_TW\\git\\data\\qrdetect',help='data set root')
    args = parser.parse_args()

    gen_train_data(args.data_dir)
