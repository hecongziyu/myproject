
import pkuseg
import logging
import os
from os.path import join
import numpy as np
import re

logger = logging.getLogger('entity')

'''
实体识别

'''
class NERSynonyms:
    '''
    基于分词器进行实体识别
    include_token指定 字性类型
    '''
    def __init__(self,dict_file,include_token=['n','v','p','b','m']):
        import synonyms
        self.include_token = include_token

    def __call__(self, text):
        tokens = synonyms.seg(text)
        tokens = self.__adjust_tokens__(tokens[0], tokens[1])
        return tokens

    def __adjust_tokens__(self, tokens, token_pos_list):
        '''
        调整token，将n开头的进行合并，并只保留include_token指定的字性类型
        '''
        a_tokens = []
        a_tokens_pos = []

        cur_pos = None
        cur_token = None
        for idx, pos in  enumerate(token_pos_list):
            if cur_pos is None:
                if pos[0] in self.include_token:
                    cur_pos = pos[0]
                    cur_token = tokens[idx]
            else:
                # if cur_pos == pos[0] or tokens[idx] == '的':
                if cur_pos == pos[0]:
                    cur_token = cur_token + tokens[idx]
                else:
                    a_tokens.append(cur_token)
                    a_tokens_pos.append(cur_pos)
                    if pos[0] in self.include_token:
                        cur_pos = pos[0]
                        cur_token = tokens[idx]
                    else:
                        cur_pos = None
                        cur_token = None
        return (a_tokens, a_tokens_pos)


class NERPkuseg:
    '''
    基于pkuseg分词器进行实体识别
    '''
    def __init__(self,dict_file, include_token=['n','v']):
        self.dict_file = dict_file
        self.seg = pkuseg.pkuseg(model_name='news', postag=True, user_dict=self.dict_file)

    def __call__(self, text):
        tokens = self.seg.cut(text)
        print('tokens :', tokens)
        return tokens

    def __adjust_tokens__(self, tokens, token_pos_list):
        '''
        调整token，将n开头的进行合并，并只保留include_token指定的字性类型
        '''
        a_tokens = []
        a_tokens_pos = []

        cur_pos = None
        cur_token = None
        for idx, pos in  enumerate(token_pos_list):
            if cur_pos is None:
                if pos[0] in self.include_token:
                    cur_pos = pos[0]
                    cur_token = tokens[idx]
            else:
                # if cur_pos == pos[0] or tokens[idx] == '的':
                if cur_pos == pos[0]:
                    cur_token = cur_token + tokens[idx]
                else:
                    a_tokens.append(cur_token)
                    a_tokens_pos.append(cur_pos)
                    if pos[0] in self.include_token:
                        cur_pos = pos[0]
                        cur_token = tokens[idx]
                    else:
                        cur_pos = None
                        cur_token = None
        return (a_tokens, a_tokens_pos)


ENTITY_UNK = 'UNK'

class NERecognition:
    '''
    NER识别，分为两种方式
    1、规则匹配
    2、深度学习（等规则匹配完成后，通过程序生成训练数据再做）

    '''
    def __init__(self,dict_file,pku_dict_file, seg_name='pkuseg', entity_aligment=None):
        '''
        dict file : 已识别完成的实体字典信息， 格式：（实体名，类别） 
        类别定义： 知识点， 其它， 无用

        '''
        self.dict_file = dict_file
        self.rec_map = {
            # 'synonyms':NERSynonyms(dict_file=self.dict_file,include_token=['n','v']),
            'pkuseg':NERPkuseg(dict_file=self.dict_file,include_token=['n'])
        }
        self.seg = self.rec_map[seg_name]
        self.entity_aligment = entity_aligment
        self.entity_map = self.__load_dict__()

    def __load_dict__(self):
        # with open(self.dict_file, 'r', encoding='utf-8') as f:
        #     txt = f.read()
        entity_map = {}
        if os.path.exists(self.dict_file):
            with open(self.dict_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for litem in lines:
                    k, v = litem.split('\t')
                    entity_map[k] = v.strip()
        return entity_map

    def __save_dict__(self):
        lines = [f'{k}\t{v}' for k,v in list(zip(self.entity_map.keys(), self.entity_map.values()))]
        lines = list(set(lines))
        with open(self.dict_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def __update_dict__(self, entity_lists):
        for entity_name, entity_pos, entity_class in entity_lists:
            if entity_class == 'OTH'  or entity_name in self.entity_map:
                pass
            else:
                self.entity_map[entity_name] = entity_class
        self.__save_dict__()


    def extract_entity(self, text):
        '''
        实体抽取, 返回实体信息，格式：（实体名，类别）
        实体抽取实现本次采用分词方式，后续可进行修改，修改成深度学习模式
        处理流程：
        1、对text文本进行分词， 取出分词后为n的词组
        2、检测词组是否为所需的词组，（本地查找 --》 baibu baike --》save local file)        
        '''
        
        # 分词
        tokens, token_pos_list = self.seg(text)
        logger.debug('ner tokens : %s ' % len(tokens))




        # 在已有词典进行查找, 对名词进行分类标注
        entity_list = [(x, token_pos_list[idx], self.entity_classify(x, token_pos_list[idx])) for idx, x in enumerate(tokens) ]
        print('entity lists :', entity_list)
        logger.debug('entity lists : %s ' % len(entity_list))


        # 更新字典信息
        self.__update_dict__(entity_list)
        return entity_list


    def entity_classify(self, entity_name, pos):
        '''
        实体类别判断，前期通过规则匹配方式，后期修改为深度学习方式
        pos 词性
        '''
        if pos != 'n':
            return 'OTH'

        entity_name = re.sub(r'[的]{0,1}性质|定义|特征','', entity_name)

        # 通过实体对齐方式，在实体分类MAP中查找
        if self.entity_aligment:
            _topk_sim_entity = self.entity_aligment.entity_similar(text1=entity_name, 
                                                                    text2_lists=self.entity_map.keys(),
                                                                    sim_threshold=0.9)
            if len(_topk_sim_entity) > 0:
                entity_name = _topk_sim_entity[0]

        # 直接在实体分类MAP中查找
        if entity_name  in self.entity_map:
            return self.entity_map[entity_name]


        # 通过爬虫查找该不识别的实体， 返回学科、人名、地名、公司、国家等分类信息, 增加到字典信息中
        # 人工标记


        return ENTITY_UNK






