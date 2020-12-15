# 生成训练数据
'''
https://github.com/asaini/Apriori  Python Implementation of Apriori Algorithm
https://medium.com/edureka/apriori-algorithm-d7cc648d4f1e    Apriori Algorithm — Know How to Find Frequent Itemsets ！！！
说明：
1、模拟生成学生在知识点表现的训练数据， 用于生成基于数据挖掘方式的知识点前置关系的关系抽取
   
   support:
    
   TID   K Scores
   1     K1(H), K2(H)    : 高分的知识组合, K1高分次数大于某值， 同时 K2高分次数大于某值   Support

   coef :
   K1(H)  -->  K2(H)  : K1知识点高分时， K2高分的置信度

   存在问题：
   1） 当知识点过多的时候， 怎么样挑选 Support, K1, K2同时高分的算法

'''
import numpy as np
from os.path import join
import logging


# 得到学生知识点表现数据
def get_student_peformance_data(data_root, file_name='StudentReponses_r.csv'):
    student_data = np.loadtxt(join(data_root, file_name), dtype=np.str, delimiter=",")
    return student_data



if __name__ == '__main__':
    import argparse
    logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                        level=logging.ERROR)
    logging.getLogger('gendata').setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r'D:\project\kgzero')
    parser.add_argument('--data_root', type=str, default=r'D:\PROJECT_TW\git\data\kg\entity\nre')

    args = parser.parse_args()

    data = get_student_peformance_data(data_root=args.data_root, file_name='StudentReponses_r.csv')
    print('data :', data[0,1::2].shape, '\n', data[0,1::2])
