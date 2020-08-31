# 增加训练数据
import os
from os.path import join
import lmdb
import cv2


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def gen_train_data(data_dir):
    env = lmdb.open(join(data_dir,'lmdb'), map_size=200511627)  
    nSamples = 0
    try: 
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))
    except Exception as e:
        pass
    key_idx = nSamples
    file_lists = os.listdir(join(data_dir,'images'))
    for idx, fitem in enumerate(file_lists):
        cache = {}
        file_name = fitem.split('.')[0]
        print('file name:', file_name)
        with open(join(data_dir, 'images', fitem),'rb') as f:
            image_data = f.read()
        cache[f'img_{key_idx}'] = image_data
        cache['total'] = str(key_idx).encode()
        writeCache(env, cache)
        key_idx = key_idx + 1




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\siamese',help='data set root')

    args = parser.parse_args()
    gen_train_data(args.data_root)