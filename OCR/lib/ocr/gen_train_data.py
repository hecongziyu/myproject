import argparse
import os
import numpy as np
import cv2
from image_utils import detect_char_area, convert_img_bin, detect_char_pos
from matplotlib import pyplot as plt
import torch
import shutil

'''
产生训练数据， 数据通过OpenCV将字符从原始图片中分开，并记录其坐标位置，然后再进行手工清洗。
'''

def clean_env(data_root, split='SampleA', dest_dir=None,background_dir='bg'):
    pass
    # dest_path = os.path.sep.join([data_root, dest_dir, split])
    # if os.path.exists(dest_path):
    #     shutil.rmtree(dest_path)
    # if not os.path.exists(dest_path):
    #     os.mkdir(dest_path)    



# 分离字符串
def split_char_image(image_file, image_height=32):
    '''
    分离原始图片，取得原始图片字符及边框，背景
    '''
    # print('image_file:', image_file)
    origin_image = cv2.imread(image_file,cv2.IMREAD_COLOR)
    image = origin_image.copy()
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray_area = image_gray.shape[0] * image_gray.shape[1]

    # 单文字块区域
    x1,y1,x2,y2 = detect_char_pos(image_gray.copy(),min_area=image_gray_area*0.05,min_y_diff=5)

    cent_y = y1 + int((y2-y1)/2)

    # 总体文字块区域，包含括号等
    # print('min area:',image_gray_area*0.001)
    bx1,by1,bx2,by2 = detect_char_area(image_gray.copy(),min_area=1,min_y_diff=5, cent_y=cent_y)

    
    

    if np.sum([x1,y1,x2,y2]) == 0 and np.sum([bx1,by1,bx2,by2]) == 0:
        raise '没有找到文字定位信息'
    char_image = image_gray.copy()[y1:y2,x1:x2]
    scale =  float(image_height / char_image.shape[0])
    char_image = cv2.resize(char_image,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    char_image = convert_img_bin(char_image,thread_pre=11)


    return origin_image, char_image, [bx1,by1,bx2,by2], [x1,y1,x2,y2]

# # 将bk image 转成二进制图片 
def convert_bk_image(image, image_height=32):
    '''
    分离原始图片，取得原始图片字符及边框，背景
    '''
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    scale =  float(image_height / image.shape[0])
    image = cv2.resize(image,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    image = convert_img_bin(image,thread_pre=11)
    return image

# 生成将答案划去的手写体字符
def gen_clean_handle_image(data_root, dest_dir):
    clean_path = os.path.sep.join([data_root,'images','CleanOrigin'])
    clean_files = os.listdir(clean_path)
    for image_file in clean_files:
        origin_image = cv2.imread(os.path.sep.join([clean_path,image_file]),cv2.IMREAD_COLOR)
        origin_image = cv2.cvtColor(origin_image,cv2.COLOR_BGR2GRAY)
        bin_image = convert_img_bin(origin_image,thread_pre=11)
        cv2.imwrite(os.path.sep.join([data_root,dest_dir,'CleanOrigin',f'{image_file}']),bin_image)







def gen_train_images(data_root, split='SampleA', background_dir='bg', image_height=32, dest_dir=None):
    split_path = os.path.sep.join([data_root,'images', split])
    files = os.listdir(split_path)
    dest_path = os.path.sep.join([data_root, dest_dir, split])
    bg_path = os.path.sep.join([data_root,dest_dir,background_dir])
    # print(files_list)
    # 将位置信息，写入文件当中, 可能会用于后期文字定位训练
    with open(os.path.sep.join([data_root, dest_dir, f'{split}_pos.txt']), 'w', encoding='utf-8') as f:    
        for idx, file_name in enumerate(files):

            try:
                # 通过Opencv文字膨胀、模糊后采用轮廓的方式取得文件区域
                # origin_image 原始图片， image 转换后的bin文字， char pos 文字在原始图片的位置信息
                origin_image, char_image, bk_pos, char_pos = split_char_image(os.path.sep.join([split_path, file_name]),image_height)
                f.write('{}|{}\n'.format(file_name, ','.join([str(x) for x in char_pos])))
                char_image = char_image.astype(np.int)
                cv2.imwrite(os.path.sep.join([dest_path,f'{file_name}']),char_image)


                #  通过上述方式，得到背景图片，用于后期训练
                bk_image =  origin_image.copy()
                # 取得背景部分，注意背景用中位数填充
                x1,y1,x2,y2 = char_pos
                bk_image[y1:y2, x1:x2, :] = np.mean(bk_image[0:y1, 0:x1,:],axis=(0,1), keepdims=True)
                bx1,by1, bx2, by2 = bk_pos
                bk_image = bk_image[by1:by2, bx1:bx2]
                # 注意这里暂时不对背景图片转换成二进制，只保存原始的， 以方便后期的其它的如位训练等
                # bk_image = convert_bk_image(bk_image.copy())

                print(os.path.sep.join([bg_path,f'bg_{file_name}']))
                cv2.imwrite(os.path.sep.join([bg_path,f'bg_{file_name}']),bk_image)                

                # plt.imshow(bk_image,'gray')
                # plt.show()

            except Exception as e:
                print(e)


# 清理垃圾数据
def clean_gen_data(data_root,dest_dir, split='SampleA'):
    '''
    清理坐标文件中错误的坐标信息
    后期，检测图片文件中像素点少于标准值范围内的图片
    '''
    sample_image_files = os.listdir(os.path.sep.join([data_root, dest_dir, split]))
    origin_image_files = os.listdir(os.path.sep.join([data_root, 'images', split]))

    all_images_files = [x for x in sample_image_files if x in origin_image_files]


    # print(all_images_files)

    pos_data_file = os.path.sep.join([data_root, dest_dir, f'{split}_pos.txt'])

    with open(pos_data_file, 'r', encoding='utf-8') as f:
        pos_data = f.readlines()

    pos_data = ['{}|{}\n'.format(x[0],x[1]) for x in [ y.strip().split('|') for y in pos_data] if x[0] in all_images_files]
    
    with open(os.path.sep.join([data_root, dest_dir, f'{split}_pos_clean.txt']), 'w', encoding='utf-8') as f:        
        f.write(''.join(pos_data))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR Evaluating Program")
    parser.add_argument('--data_root',default='D:\\PROJECT_TW\\git\\data\\ocr', type=str, help='path of the evaluated model')
    parser.add_argument('--split', default='SampleA')
    parser.add_argument("--image_height", type=int, default=32, help="图片高度")                
    parser.add_argument("--tmp_dir", type=str, default='tmp', help="临时目录, 用于测试用")   
    parser.add_argument('--dest_dir',type=str, default='dest', help='原始文件处理后另存的目录')             
    args = parser.parse_args()
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    # clean_env(args.data_root, args.split, args.dest_dir)
    # splits = ['SampleA','SampleB','SampleC','SampleD','SampleE','SampleH','SampleZ']
    splits = ['SampleA','SampleB','SampleC','SampleD','SampleE','SampleF','SampleG','SampleH']
    # for split in splits:
    #     gen_train_images(data_root=args.data_root, split=split, 
    #                     image_height=args.image_height,dest_dir=args.dest_dir)

    # for split in splits:
    #     clean_gen_data(data_root=args.data_root, dest_dir=args.dest_dir, split=split)

    # gen_clean_handle_image(data_root=args.data_root, dest_dir=args.dest_dir)