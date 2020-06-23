# -*- coding: UTF-8 -*-
import os
from txt_utils import gen_question_no
from txtdataset import TAG_QUESTION, TAG_CONTENT, TAG_ANSWER
'''
数据预处理
1、question.txt,  title.txt,  content.txt 进行合并， 并标注 Label
2、用于生成分词的词向量 vocab
'''

def merge_and_label(data_root, size=1000):
    '''
    合并文档，并做标注
    '''

    with open(os.path.sep.join([data_root,'source','random.txt']), 'r', encoding='utf-8') as f:
        random_txt = f.readlines()

    with open(os.path.sep.join([data_root,'source','content.txt']), 'r', encoding='utf-8') as f:
        content_txt = f.readlines()

    with open(os.path.sep.join([data_root,'source','answer.txt']),'r', encoding='utf-8') as f:
        answer_txt = f.readlines()

    qn_lists = gen_question_no()

    paper_txts = []

    paper_txts.extend([f'{x.strip()}|{TAG_QUESTION}' for x in random_txt])
    paper_txts.extend([f'{x.strip()}|{TAG_CONTENT}' for x in random_txt ])
    paper_txts.extend([f'{x.strip()}|{TAG_CONTENT}' for x in content_txt ])
    paper_txts.extend([f'{x.strip()}|{TAG_ANSWER}' for x in answer_txt ])

    with open(os.path.sep.join([data_root,'train.txt']), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in paper_txts])

    print('生成训练文件完成， 保存目录{}, 共{}条记录'.format(os.path.sep.join([data_root,'train.txt']), len(paper_txts)))
    return paper_txts




def gen_custom_dict(data_root):
    '''
    生成分词 自有分词字典
    '''

    qn_lists = gen_question_no()
    custom_str = '填空题,选择题,解答题'
    qn_lists.extend(custom_str.split(','))

    qn_lists = set([x.replace('（','(').replace('）',')').replace('、','.').replace(',','.') for x in qn_lists])
    # print(qn_lists)
    


    with open(os.path.sep.join([data_root,'use_dict.txt']), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in qn_lists])



    print('生成分词字典完成')


def load_custom_dict(data_root):
    with open(os.path.sep.join([data_root,'use_dict.txt']), 'r', encoding='utf-8') as f:
        lexicon = f.readlines()

    lexicon = [x.strip() for x in lexicon]
    return lexicon







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test paper dataset')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\testpaper',help='data set root')
    args = parser.parse_args()
    # merge_and_label(data_root=args.data_root)
    gen_custom_dict(data_root=args.data_root)


