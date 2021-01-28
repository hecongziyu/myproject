# 增加训练数据
# https://www.pianshen.com/article/655878816/ 手工标注工具
# https://blog.csdn.net/xunan003/article/details/78720189/
import os
from os.path import join
import lmdb
import cv2
import numpy as np
from matplotlib import pyplot as plt
import shutil
from xml.dom.minidom import parse


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def gen_train_data(data_dir):
    env = lmdb.open(join(data_dir,'lmdb'), map_size=3800511627)  
    qr_nSamples = 0
    bg_nSamples = 0
    try: 
        with env.begin(write=False) as txn:
                qr_nSamples = int(txn.get('qr_total'.encode()))
    except Exception as e:
        pass

    try: 
        with env.begin(write=False) as txn:
            bg_nSamples = int(txn.get('bg_total'.encode()))
    except Exception as e:
        pass        


    q_key_idx = 0
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


    # b_key_idx = bg_nSamples
    # file_lists = os.listdir(join(data_dir,'images', 'bg'))


    # for idx, fitem in enumerate(file_lists):
    #     cache = {}
    #     file_name = fitem.split('.')[0]
    #     # print('file name:', file_name)
    #     # with open(join(data_dir, 'images','bg',fitem),'rb') as f:
    #     #     image_data = f.read()
    #     print('file :', join(data_dir,'images','bg',fitem))
    #     image = cv2.imread(join(data_dir,'images','bg',fitem), cv2.IMREAD_COLOR)
    #     heigh, width, _ = image.shape
    #     radio = 2048/heigh
    #     image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
    #     _, image_data = cv2.imencode(".png", image)

    #     cache[f'bg_{b_key_idx}'] = image_data
    #     cache[f'target_{b_key_idx}'] = np.array([[-1,-1,-1,-1,-1]],dtype=np.float).tobytes(order='C')
    #     cache['bg_total'] = str(b_key_idx).encode()
    #     writeCache(env, cache)
    #     b_key_idx = b_key_idx + 1  

# 根据真实环境生成图片信息，包含二维码位置信息
def gen_real_train_data(data_dir):

    env = lmdb.open(join(data_dir,'lmdb'), map_size=3800511627)  
    qr_nSamples = 0
    bg_nSamples = 0
    try: 
        with env.begin(write=False) as txn:
                qr_nSamples = int(txn.get('qr_total'.encode()))
    except Exception as e:
        pass

    try: 
        with env.begin(write=False) as txn:
            bg_nSamples = int(txn.get('bg_total'.encode()))
    except Exception as e:
        pass        

    print('qr samples:', qr_nSamples, ' bg samples :', bg_nSamples)

    qr_img_lists = []
    with open(join(data_dir,'images','real','qr_pos.txt'), 'r', encoding='utf-8') as f:
        qr_lists = f.readlines()

        for q in qr_lists:
            img_file, img_qr = q.strip().split('||')
            if os.path.exists(join(data_dir,'images','real','source',img_file)) and \
                os.path.exists(join(data_dir, 'images','real','result', img_file)):
                qr_pos_lists = eval(img_qr)
                qr_img_lists.append((img_file, qr_pos_lists))
            else:
                # print(img_file , ' is not correct')
                pass


    print('qr_img_lists:', qr_img_lists[0], ' len lists :', len(qr_img_lists))

    # b_key_idx = bg_nSamples
    b_key_idx = 42

    for _file, _pos in qr_img_lists:
        cache = {}
        print('add img to lmdb :', b_key_idx)
        image = cv2.imread(join(data_dir,'images','real','source',_file), cv2.IMREAD_COLOR)
        heigh, width, _ = image.shape
        radio = 2048/heigh
        image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
        _, image_data = cv2.imencode(".png", image)

        pos_array = np.array(_pos)
        pos_array = pos_array * radio
        pos_array = pos_array.reshape(-1,4)
        pos_array = np.c_[pos_array,np.zeros(4)]
        pos_array = pos_array.astype(np.float)
        # for box in pos_array:
        #     x0, y0, x1, y1, label= box
        #     print('box :', box, ' x0:', x0)
        #     if int(label) == 0:
        #         cv2.rectangle(image, (x0,y0), (x1, y1), (0, 255, 0), 2)

        # plt.imshow(image)
        # plt.show()

        cache[f'bg_{b_key_idx}'] = image_data
        cache[f'target_{b_key_idx}'] = pos_array.tobytes(order='C')
        cache['bg_total'] = str(b_key_idx).encode()
        writeCache(env, cache)
        b_key_idx = b_key_idx + 1  


# 根据真实环境生成二维码标记
def gen_real_qr_image(data_dir):

    env = lmdb.open(join(data_dir,'lmdb'), map_size=3800511627)  
    qr_nSamples = 0
    bg_nSamples = 0
    try: 
        with env.begin(write=False) as txn:
                qr_nSamples = int(txn.get('qr_total'.encode()))
    except Exception as e:
        pass

    try: 
        with env.begin(write=False) as txn:
            bg_nSamples = int(txn.get('bg_total'.encode()))
    except Exception as e:
        pass        

    # print('qr samples:', qr_nSamples, ' bg samples :', bg_nSamples)

    qr_img_lists = []
    with open(join(data_dir,'images','real','qr_pos.txt'), 'r', encoding='utf-8') as f:
        qr_lists = f.readlines()

        for q in qr_lists:
            img_file, img_qr = q.strip().split('||')
            if os.path.exists(join(data_dir,'images','real','source',img_file)) and \
                os.path.exists(join(data_dir, 'images','real','result', img_file)):
                qr_pos_lists = eval(img_qr)
                qr_img_lists.append((img_file, qr_pos_lists))
            else:
                # print(img_file , ' is not correct')
                pass


    print('qr_img_lists:', qr_img_lists[0], ' len lists :', len(qr_img_lists))
    qr_idx = qr_nSamples + 1

    for _file, _pos in qr_img_lists:
        cache = {}
        
        image = cv2.imread(join(data_dir,'images','real','source',_file), cv2.IMREAD_COLOR)
        heigh, width, _ = image.shape
        radio = 2048/heigh
        image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
        _, image_data = cv2.imencode(".png", image)

        pos_array = np.array(_pos)
        pos_array = pos_array * radio
        pos_array = pos_array.reshape(-1,4)
        pos_array = np.c_[pos_array,np.zeros(4)]
        pos_array = pos_array.astype(np.int)

        for box in pos_array:
            if np.random.randint(3) == 0:
                x0, y0, x1, y1, label= box
                qr_img = image[y0:y1, x0:x1, :]
                cv2.imwrite(join(data_dir, 'images', 'qr', f'q{qr_idx}.png'), qr_img)
                print('gen qr image :', qr_idx)
                qr_idx = qr_idx + 1


# 错误识别图形手工处理预处理部分，将错误识别原图复制到对应目录里面准备手工处理
def pre_handle_error(data_dir):
    e_file_lists = os.listdir(os.path.join(data_dir, 'error_imgs', 'error'))
    print('e file lists :', e_file_lists)
    # 检测该图片是否已在训练数据中
    for eitem in e_file_lists:
        # eitem = eitem.rsplit('.',1)[0]
        # eitem = eitem.replace('.png','')

        if os.path.exists(join(data_dir, 'result', eitem)) or os.path.exists(join(data_dir, 'error_imgs', 'taged', eitem)):
            print('文件已在训练数据中, ', eitem)
        else:
            print('将原始文件添加到手工处理目录中:', eitem)
            image = cv2.imread(join(data_dir,'images','real','source',eitem),cv2.IMREAD_COLOR)
            radio = 2048/image.shape[0]
            image = cv2.resize(image, (0,0), fx=radio, fy=radio, interpolation=cv2.INTER_AREA)
            cv2.imwrite(join(data_dir, 'error_imgs', 'source',eitem), image)
            # shutil.copy(join(data_dir,'images', 'real','source', eitem), join(data_dir, 'error_imgs', 'source'))


# 生成错误识别后手工标记数据
def gen_handle_qr_image(data_dir):
    env = lmdb.open(join(data_dir,'lmdb'), map_size=3800511627)  
    qr_nSamples = 0
    bg_nSamples = 0
    try: 
        with env.begin(write=False) as txn:
                qr_nSamples = int(txn.get('qr_total'.encode()))
    except Exception as e:
        pass

    try: 
        with env.begin(write=False) as txn:
            bg_nSamples = int(txn.get('bg_total'.encode()))
    except Exception as e:
        pass        

    print('qr samples:', qr_nSamples, ' bg samples :', bg_nSamples)


    file_lists = os.listdir(join(data_dir, 'error_imgs', 'source'))
    file_lists = [x for x in file_lists if x.find('.xml') != -1]

    for item in file_lists:
        domtree = parse(join(data_dir,'error_imgs', 'source',item))
        imgdom = domtree.documentElement
        img_name = imgdom.getElementsByTagName('filename')[0].firstChild.nodeValue
        print('img name :', img_name)
        image = cv2.imread(join(data_dir,'error_imgs', 'source', img_name), cv2.IMREAD_COLOR)
        print('image shape :', image.shape)
        # image = image.astype(np.uint8)
        box_lists = []
        box_nodes = imgdom.getElementsByTagName('bndbox')
        print('box_lists :',  len(box_nodes))
        for box in box_nodes:
            xmin = int(box.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            ymin = int(box.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            xmax = int(box.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            ymax = int(box.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            box_lists.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin,ymin), (xmax, ymax), (0, 255, 0), 5)


            # _pos = [box.childNodes[0].childNodes[0].nodeValue,box.childNodes[1].nodeValue,box.childNodes[2].nodeValue,box.childNodes[3].nodeValue]
            # _pos = [x.strip() for x in _pos]            
            # box_lists.append(_pos)
        # plt.imshow(image)
        # plt.show()
        cv2.imwrite(join(data_dir,'valid_imgs',f'{item}.png'), image)

        print('box lists :', box_lists)

        # with open(join(data_dir,'error_imgs', 'taged',item), 'r', encoding='utf-8') as f:
        #     text = f.read()
        #     print('text :', text)
        # break







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='gen latex pos train data')
    parser.add_argument('--data_dir', default='D:\\PROJECT_TW\\git\\data\\qrdetect',help='data set root')
    args = parser.parse_args()

    # 生成训练图片
    # gen_train_data(args.data_dir)

    # 通过真实环境下在图片生成训练图片
    # gen_real_train_data(args.data_dir)

    # 通过真实环境下的图片生成QR图片，用于训练
    # gen_real_qr_image(args.data_dir)


    # 错误识别图形手工处理预处理部分
    pre_handle_error(args.data_dir)

    # 生成错误识别后手工标记数据
    # gen_handle_qr_image(args.data_dir)