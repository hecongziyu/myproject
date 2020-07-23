# -*- coding: UTF-8 -*-
from collections import Counter
from itertools import product
import Levenshtein
import re
import numpy as np





# 生成题号
def gen_question_no():
    question_no= [list(range(1,50)), 
                        '一,二,三,四,五,六,七,八,九,十'.split(','),
                        # 'A,B,C,D,E,F,G'.split(','),
                        'I,II,III'.split(','),
                        'Ⅰ,Ⅱ,Ⅲ,i,ii,iii,ⅰ,ⅱ'.split(',')]
    # 不需加标点符号
    # querstion_no_special = '①,②,③,④,⑤,⑥,⑦,⑧,⑨,⑩,⑴,⑵'.split(',')
    # [OL] 针对HTML中没有<ol>标记，该标记不带题号   [REV] 保留
    querstion_no_special = '[OL],[REV]'.split(',')
    # 标点符号
    punctuation = ['.',')','题',('(',')')]

    qn_lists = []

    for qls in question_no:
        for qitem, pitem in product(qls, punctuation):
            if type(pitem) == tuple:
                q_n = '{}{}{}'.format(pitem[0], qitem, pitem[1])
            else:
                q_n = f'{qitem}{pitem}'
            qn_lists.append(q_n)

    qn_lists.extend(querstion_no_special)
    # print(len(qn_lists))
    return qn_lists

def gen_question_no_type():
    qn_type = {}
    qn_type['TYPE_A'] = '一,二,三,四,五,六,七,八,九,十'.split(',')
    qn_type['TYPE_B'] = [str(x) for x in list(range(1,50))]
    # qn_type['TYPE_C'] = '①,②,③,④,⑤,⑥,⑦,⑧,⑨,⑩'.split(',')
    qn_type['TYPE_C'] = ['{})'.format(x) for x in list(range(1,50))]
    qn_type['TYPE_D'] = 'Ⅰ,Ⅱ,Ⅲ,I,II,III'.split(',')
    qn_type['TYPE_E'] = 'i,ii,ⅰ,ⅱ'.split(',')
    # qn_type['TYPE_F'] = 'I,II,III'.split(',')
    qn_type['TYPE_SPECIAL'] = '[OL],[REV]'.split(',')
    return qn_type





# 移除标点符号
def remove_puct(text):
    return text.replace('(','').replace('（','').replace(')','').replace('）','').replace('.','').replace(',','').replace('，','')

def replace_content(text):
    result = re.sub(r'\{img:\d+\}','', text)
    result = result.replace('\n','').replace('[OL]','').replace(' ','').replace('　','').strip()
    return result

# 比较两句话的相似性
def txt_ratio(txt_1, txt_2):
    txt_1 = replace_content(txt_1)
    txt_2 = replace_content(txt_2)
    if len(txt_1) < 2 or len(txt_2) < 2:
        return 0
    else:
        return Levenshtein.jaro(txt_1, txt_2)


# 合并两个字符串， img_str中包括{img:11}这样类似的图片信息，需将该图片信息插入到sim_str里面对应的位置
def combine_include_img_str(img_str, sim_str):
    result = []
    # 注意img_str需去掉空格，防止空格参与匹配， sim str不以去掉
    img_str = img_str.replace('[OL]','').replace(' ','').replace('　','')
    img_lists = re.findall(r'\{img:\d+\}', img_str)
    if len(img_lists) == 0:
        return sim_str

    # 定位不包含{img}字符的开始位置
    no_img_str =  re.sub(r'\{img:\d+\}','',img_str)
    no_img_pos_begin = img_str.find(no_img_str[0])

    # print('no img str:', no_img_str, ' no img pos begin:', no_img_pos_begin)

    # 注意长度需-1， 因为后面定位时，定位位置是从0开始的, 采用cumsum会多累加一位的长度
    img_len_lists = [len(x) - 1 for x in img_lists]
    img_len_lists.insert(0,0)
    # print('befor cosume:', img_len_lists)
    img_len_lists = np.cumsum(img_len_lists).tolist()

    no_img_pos_lists = [no_img_pos_begin - img_len_lists[idx] for idx,x in enumerate(img_lists)][1:]
    no_img_pos_lists = [x for x in no_img_pos_lists if x > 0]
    no_img_pos_begin = min(no_img_pos_lists) if len(no_img_pos_lists) > 0 else -1

    # {img:xx} 在字符串的位置
    img_pos_lists = [img_str.find(x)  for idx,x in enumerate(img_lists)]
    # print('before img_pos_lists', img_pos_lists)
    # print('image len lists:', img_len_lists)
    img_pos_lists = [img_str.find(x)-img_len_lists[idx] for idx,x in enumerate(img_lists)]
    
    

    # 包含{img:xx}的字符串，去掉{img:xx}， 然后依次与 sim_str进行比较
    img_str_lists = list(re.sub(r'\{img:\d+\}','',img_str))
    sim_str_lists = list(sim_str)
    
    cur_sim_pos = 0
    cur_img_str_pos = 0

    # 记录第一个匹配的位置
    _first_sim_pos = 0

    # print('after img pos lists:', img_pos_lists)
    # print('no img begin pos lists:', no_img_pos_lists, ' no img pos begin :', no_img_pos_begin)
    # print('sim_str_lists:', sim_str_lists)
    # print('img_str_lists:', img_str_lists)


    _pos_is_zero_len = len([x for x in img_pos_lists if x < no_img_pos_begin])

    result = list(sim_str[sim_str.find(img_str_lists[0]):])
    for img_idx, img_pos in enumerate([x for x in img_pos_lists if x > no_img_pos_begin]):
        result.insert(img_pos, img_lists[img_idx+_pos_is_zero_len])

    
    for idx in range(_pos_is_zero_len):
        if _first_sim_pos != 0:
            result.insert(_first_sim_pos-1, img_lists[_pos_is_zero_len-idx-1])
        else:
            result.insert(0, img_lists[_pos_is_zero_len-idx-1])

    result = ''.join(sim_str[0:sim_str.find(img_str_lists[0])]) + ''.join(result)

    return result

def combine_include_img_str_backup(img_str, sim_str):
    result = []
    # 注意img_str需去掉空格，防止空格参与匹配， sim str不以去掉
    img_str = img_str.replace('[OL]','').replace(' ','').replace('　','')
    img_lists = re.findall(r'\{img:\d+\}', img_str)
    if len(img_lists) == 0:
        return sim_str

    # 定位不包含{img}字符的开始位置
    no_img_str =  re.sub(r'\{img:\d+\}','',img_str)
    no_img_pos_begin = img_str.find(no_img_str[0])

    print('no img str:', no_img_str, ' no img pos begin:', no_img_pos_begin)

    # 注意长度需-1， 因为后面定位时，定位位置是从0开始的, 采用cumsum会多累加一位的长度
    img_len_lists = [len(x) - 1 for x in img_lists]
    img_len_lists.insert(0,0)
    # print('befor cosume:', img_len_lists)
    img_len_lists = np.cumsum(img_len_lists).tolist()

    no_img_pos_lists = [no_img_pos_begin - img_len_lists[idx] for idx,x in enumerate(img_lists)][1:]
    no_img_pos_lists = [x for x in no_img_pos_lists if x > 0]
    no_img_pos_begin = min(no_img_pos_lists) if len(no_img_pos_lists) > 0 else -1

    # {img:xx} 在字符串的位置
    img_pos_lists = [img_str.find(x)  for idx,x in enumerate(img_lists)]
    # print('before img_pos_lists', img_pos_lists)
    # print('image len lists:', img_len_lists)
    img_pos_lists = [img_str.find(x)-img_len_lists[idx] for idx,x in enumerate(img_lists)]
    
    

    # 包含{img:xx}的字符串，去掉{img:xx}， 然后依次与 sim_str进行比较
    img_str_lists = list(re.sub(r'\{img:\d+\}','',img_str))
    sim_str_lists = list(sim_str)
    
    cur_sim_pos = 0
    cur_img_str_pos = 0

    # 记录第一个匹配的位置
    _first_sim_pos = 0

    print('after img pos lists:', img_pos_lists)
    print('no img begin pos lists:', no_img_pos_lists, ' no img pos begin :', no_img_pos_begin)
    # print('sim_str_lists:', sim_str_lists)
    # print('img_str_lists:', img_str_lists)


    _pos_is_zero_len = len([x for x in img_pos_lists if x < no_img_pos_begin])
    _last_img_pos = 0

    for img_idx, img_pos in enumerate([x for x in img_pos_lists if x > no_img_pos_begin]):
        # 检测当前sim 偏移位置是否已大于 sim str长度， 如果超过，证明sim str已匹配完
        # img str: 当时，{img:0}A{img:1} {img:2}
        # sim str: (1) 当时，A
        # img pos lists: [3, 4, 5]
        # sim_str_lists: ['(', '1', ')', ' ', '当', '时', '，', 'A']
        # img_str_lists: ['当', '时', '，', 'A', ' ']
        print('pos:', img_pos, ' cur sim pos:', cur_sim_pos, ' cur_img_str_pos:',cur_img_str_pos,'  len:', len(sim_str_lists))
        if cur_sim_pos < len(sim_str_lists) :
            for sidx, _simchar in enumerate(sim_str_lists[cur_sim_pos:]):
                for _iidx, _imgstrchar in enumerate(img_str_lists[cur_img_str_pos:]):

                    if _simchar == _imgstrchar:
                        if img_idx == 0:
                            _first_sim_pos = sidx
                            # print('IS OK:', _simchar, ':', _imgstrchar)
                        cur_img_str_pos = _iidx + cur_img_str_pos + 1
                        break

                # print('------------cur img str pos:', cur_img_str_pos, ' char:', _simchar, ' img pos:', img_pos, ' last img pos:', _last_img_pos)
                result.append(_simchar)
                # 检测该字符串后面的定位位置是否与图片位置相同
                if cur_img_str_pos == img_pos :
                    cur_sim_pos = sidx + cur_sim_pos + 1
                    _last_img_pos = img_pos
                    # cur_img_str_pos = cur_img_str_pos + 1
                    result.append(img_lists[img_idx+_pos_is_zero_len])
                    break
        else:
            result.append(img_lists[img_idx+_pos_is_zero_len])

    result.extend(sim_str_lists[cur_sim_pos:])

    # 处理img_pos[0] 的位置为0的情况
    
    for idx in range(_pos_is_zero_len):
        if _first_sim_pos != 0:
            result.insert(_first_sim_pos-1, img_lists[_pos_is_zero_len-idx-1])
        else:
            result.insert(0, img_lists[_pos_is_zero_len-idx-1])
    return ''.join(result)


def load_vocab(data_dir):
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))
    return vocab


if __name__ == '__main__':
    # combine_include_img_str('')
    pass