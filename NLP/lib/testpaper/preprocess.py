# -*- coding: UTF-8 -*-
import os
from utils.txt_utils import gen_question_no
from txtdataset import TAG_QUESTION, TAG_CONTENT, TAG_ANSWER,TAG_ANSWER_AREA, STOP_WORDS
import pkuseg
import torchtext.data as data
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

'''
数据预处理
1、question.txt,  title.txt,  content.txt 进行合并， 并标注 Label
2、用于生成分词的词向量 vocab
'''



def merge_and_label(data_root, size=1000):
    '''
    合并文档，并做标注
    '''

    with open(os.path.sep.join([data_root,'source','question.txt']), 'r', encoding='utf-8') as f:
        random_txt = f.readlines()

    with open(os.path.sep.join([data_root,'source','content.txt']), 'r', encoding='utf-8') as f:
        content_txt = f.readlines()

    with open(os.path.sep.join([data_root,'source','answer.txt']),'r', encoding='utf-8') as f:
        answer_txt = f.readlines()

    with open(os.path.sep.join([data_root,'source','answer_area.txt']),'r', encoding='utf-8') as f:
        answer_area_txt = f.readlines()

    qn_lists = gen_question_no()

    paper_txts = []

    paper_txts.extend([f'{x.strip()}|{TAG_QUESTION}' for x in random_txt])
    paper_txts.extend([f'{x.strip()}|{TAG_CONTENT}' for x in random_txt ])
    paper_txts.extend([f'{x.strip()}|{TAG_CONTENT}' for x in content_txt ])
    paper_txts.extend([f'{x.strip()}|{TAG_ANSWER}' for x in answer_txt ])
    paper_txts.extend([f'{x.strip()}|{TAG_ANSWER_AREA}' for x in answer_area_txt ])

    with open(os.path.sep.join([data_root,'train.txt']), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in paper_txts])

    with open(os.path.sep.join([data_root,'valid.txt']), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in paper_txts])

    print('生成训练文件完成， 保存目录{}, 共{}条记录'.format(os.path.sep.join([data_root,'train.txt']), len(paper_txts)))
    return paper_txts




def gen_custom_dict(data_root):
    '''
    生成分词 自有分词字典
    '''

    qn_lists = gen_question_no()
    custom_str = '''填空题,选择题,img,解析,解法,答案,求,解答,过程,步骤,注意,考生,高等,学校,试题,
                    参考答案,点评,解答题,分析,选做题,证明,考试,第一,第二, 第三,部分,文科,
                    数学,理科,小时,面积,函数,速度,范围,令,验证,要求,正确,错误,是否, 理由,
                    化学,计算,若,在,如,图,意思,充分,必要,向量,元素,个数,因为,所以,故,显然,
                    小题,下图,命题,若,设,已知,满分,解:,解：,同理,方程,可证,分析,思路,基础,知识,
                    定理,证法,如图,关系,答题卡,答题,观察,说明,文字,满分,分,于是,根据,则,
                    概念,考点,性质,图,因此,综合,化学,实验,选项,其余,题号,其它,给分,检测,
                    模块,期末,期中,选,统一,注意,事项,姓名,座号,规定,涂,改动,必须,签字,2B,
                    铅笔,标号,填写,参考,直接,第,卷,每,右,左,满分,容器,坐标,普通,草稿,橡皮,
                    改动,整洁,结束,保持,中学,确定,必做题,汇总,成立,答,不,下列,结束,点拨,
                    精讲,精析,命题,意图,评分,标准,解析式,答案汇总,评分标准,与,共,本部分,
                    ,每小题,无解,】,【,:,：,∴,∵'''
    alpha_lists = ['A.', 'A)', '(A)', 'B.', 'B)', '(B)', 'C.', 'C)', '(C)', 'D.', 'D)', '(D)', 'E.', 'E)', '(E)', 'F.', 'F)', '(F)', 'G.', 'G)', '(G)']
    qn_lists.extend(custom_str.split(','))
    qn_lists.extend(alpha_lists)

    qn_lists = set([x.replace('（','(').replace('）',')').replace('、','.').replace(',','.').strip() for x in qn_lists])
    qn_lists = sorted(qn_lists)

    print(qn_lists)

    with open(os.path.sep.join([data_root,'weights','use_dict.txt']), 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in qn_lists])

    print('生成分词字典完成')



# 注意不能用extend方式，需采用append, 结果为 [['1'],['2']], 这样的结果，否则会有问题
# collections  Counter  计算方式引起
def tokenizer_lists(seg, lines,lexicon):  
    def filter(word):
        if word in lexicon:
            return True
        return False

    tokens = []
    for l in lines:
        tokens.append([wd for wd in seg.cut(l) if filter(wd)])
    return tokens


# def build_vocab(data_root):
#     '''
#     创建词向量表
#     '''

#     # 初始化分词工具
#     lexicon = load_custom_dict(data_root)
#     lexicon.sort()
#     seg = pkuseg.pkuseg(user_dict=lexicon) 

#     TEXT = data.Field(tokenize=None,lower=False, batch_first=True, postprocessing=None,stop_words=STOP_WORDS)


#     # file_lists = os.listdir(os.path.sep.join([data_root, 'output']))
#     file_contents = []
#     file_contents.append(lexicon)


#     print('tokens:', len(file_contents))
#     TEXT.build_vocab(file_contents,min_freq=1)

#     print(TEXT.vocab.itos)
#     print(len(TEXT.vocab))
#     return TEXT.vocab
    

def gen_train_file(data_root):

    def replace_content(text, replace_lists):
        text = text.strip()
        text = text.replace('．','.').replace('（','(').replace('）',')').replace('、','.')
        for item in replace_lists:
            text = text.replace(item, '')
        return text

    question_no = gen_question_no()
    question_no = sorted(question_no, key=lambda x:len(x), reverse=True)
    print(question_no)

    filter_no = ['A.', 'A)', '(A)', 'B.', 'B)', '(B)', 'C.', 'C)', '(C)', 'D.', 'D)', '(D)', 'E.', 'E)', '(E)', 'F.', 'F)', '(F)', 'G.', 'G)', '(G)']

    question_no = [x for x in question_no if x not in filter_no]


    file_lists = os.listdir(os.path.sep.join([data_root, 'output']))
    file_contents = []
    for idx, file_name in enumerate(file_lists):
        with open(os.path.sep.join([data_root,'output', file_name]), 'r', encoding='utf-8') as f:
            file_contents.extend(f.readlines())
    print(file_contents[0:40])

    file_contents = ['%s\n' % replace_content(x, question_no) for x  in file_contents if len(x) > 1]
    print('-----------------------------------------------')
    print(file_contents[0:30])
    with open(os.path.sep.join([data_root,'source','all.txt']), 'w', encoding='utf-8') as f:
        f.writelines(file_contents)

    print('生成全量文件完成 ', len(file_contents))



def load_custom_dict(data_root):
    with open(os.path.sep.join([data_root,'weights','use_dict.txt']), 'r', encoding='utf-8') as f:
        lexicon = f.readlines()
    lexicon = [x.strip() for x in lexicon]
    lexicon = list(set(lexicon))
    lexicon.sort()
    return lexicon







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test paper dataset')
    parser.add_argument('--data_root', default='D:\\PROJECT_TW\\git\\data\\testpaper',help='data set root')
    args = parser.parse_args()
    # merge_and_label(data_root=args.data_root)
    gen_custom_dict(data_root=args.data_root)
    # merge_and_label(data_root=args.data_root)
    # build_vocab(data_root=args.data_root)
    # gen_train_file(data_root=args.data_root)


