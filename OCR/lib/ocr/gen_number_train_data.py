# -*- coding:utf-8 -*-
# 通过PDF，生成训练数据
# !!!! 生成Latex公式  https://mbd.baidu.com/newspage/data/landingsuper?context=%7B%22nid%22%3A%22news_9166116218594072791%22%7D&n_type=0&p_from=1

import os
import argparse
import numpy as np
import random
from tools import pdf_latex_utils as pdf
from tools import latex_txt_utils as txt
import cv2
import lmdb
import time


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def write_error(data_root, err_lines):
    with open(os.path.sep.join([data_root, 'formula_error.txt']), 'a', encoding='utf-8') as f:
        err_lines = '\n'.join(err_lines)
        f.write(err_lines)


# 生成训练数据, 通过latex组成pdf
''' https://latex.91maths.com/eg/czjxjh.html 常用的学校数学公式
    data root :     数据根目录
    formula file:   数学公式文本  
'''
def gen_train_data_by_pdf(data_root, formula_file, begin_pos=8750, key_idx=6586, size=100):

    outputPath = os.path.sep.join([data_root, 'lmdb'])
    env = lmdb.open(outputPath, map_size=10511627)

    with open(os.path.sep.join([data_root,formula_file]), 'r', encoding='utf-8') as f:
        formula_lines = f.readlines()

    print('formula lines:', len(formula_lines))
    # 不支持数组方式，后期看怎么解决
    # formula_lines = [txt.latex_remove_space(x).strip() for x in formula_lines if len(x.strip()) > 0]

    key_idx = 0
    try:
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))    

        key_idx = nSamples 
    except:
        pass

    # key_idx = 3500

    print('key idx:', key_idx)

    batch_size = 5
    m_lines = []
    start_time = time.time()

    for idx, m in enumerate(formula_lines[begin_pos:]):
        if idx % 50 ==  0:
            print('已经处理{}条数据, 索引序列: {} 耗时：{:.3f}'.format(idx,key_idx,(time.time() - start_time)))
            start_time = time.time()

        if (idx+1) % batch_size == 0:

            try:
                # 通过PDF生成数学公式图片
                print('m lines:', m_lines)
                pdf.gen_latex_pdf(data_root, 'temp', m_lines)        
                # 得到PDF中公式图片位置信息
                math_images = pdf.gen_latex_img_pos(data_root=data_root, file_name='temp',imgH=8192)         
                print('idx :', idx, ' math images len:', len(math_images))   

                cache = {}

                for _idx in range(len(m_lines)):
                    # t_key =
                    cache[f't_{key_idx}'] = m_lines[_idx].encode()
                    cache[f'i_{key_idx}'] = cv2.imencode('.jpg', math_images[_idx])[1].tobytes()
                    cache['total'] = str(key_idx).encode()
                    key_idx = key_idx + 1

                writeCache(env,cache)
            except Exception as e:
                print(e)
                write_error(data_root, m_lines)

            m_lines = []
        else:
            m_lines.append(m)
        
        # writeCache(env, {'total':str(key_idx).encode()})
    
    print('生成训练数据完成！')



# 生成训练数据，采用下载的训练数据, 暂时不用。
def gen_train_data_by_file(data_root):
    formula_file_name = os.path.sep.join([data_root, 'data', 'im2latex_formulas.norm.txt'])
    train_file_name = os.path.sep.join([data_root, 'data','im2latex_train_filter.txt'])

    with open(formula_file_name, 'r', encoding='utf-8') as f:
        formula_lines = f.readlines()

    with open(train_file_name, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    outputPath = os.path.sep.join([data_root, 'lmdb'])
    env = lmdb.open(outputPath, map_size=5024511627)        


    for idx, item in enumerate(train_lines):
        img_name, formula_idx = item.split()
        # print('image name:', img_name, ' formula index:', formula_idx)
        image_data = get_image_data(os.path.sep.join([data_root,'data', 'images',img_name]))
        formula_txt = formula_lines[int(formula_idx)]
        cache = {}
        cache[f't_{idx}'] = formula_txt.encode()
        cache[f'i_{idx}'] = image_data
        if idx % 100 == 0:
            cache['total'] = str(idx).encode()
            print('load data number : ', idx)
        writeCache(env, cache)
    cache = {}
    cache['total'] = str(idx).encode()
    writeCache(env, cache)



def get_image_data(image_file):
    with open(image_file, 'rb') as f:
        data = f.read()
    return data




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\ocr\\number',help='data set root')
    parser.add_argument('--formula_file', default='im2latex_formulas_custom.txt', type=str, help='数学公式文本')
    parser.add_argument('--gen_size', default='5',type=int,help='gen size')    
    args = parser.parse_args()
    # gen_train_data_by_file(args.data_root)
    gen_train_data_by_pdf(args.data_root,formula_file=args.formula_file, begin_pos=0,key_idx=0)