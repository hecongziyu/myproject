# -*- coding: UTF-8 -*-
from model import *
# from txtdataset import TARGETS,STOP_WORDS,TAG_QUESTION,TAG_CONTENT,TAG_ANSWER,TAG_ANSWER_AREA
from config import _C as cfg, logger
import pkuseg
import numpy as np
from utils.txt_utils import gen_question_no,gen_question_no_type
import pickle
import os
import torch
import re

# 题号信息
qn_lists = gen_question_no()
qn_type_map = gen_question_no_type()

TAG_QUESTION = 1
TAG_CONTENT = 2
TAG_ANSWER = 3
TAG_ANSWER_AREA = 4

Q_TYPE_SELECT = 1
Q_TYPE_EMPTY = 2
Q_TYPE_QA = 3
Q_TYPE_UNK = -1

TARGETS = {1:'QUERSTION', 2:'CONTENT', 3:'ANSWER', 4:'ANSWER_AREA'}


class PaperDetect(object):
    def __init__(self):
        self.use_cuda = True if cfg.paper.model.use_cuda and torch.cuda.is_available() else False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.vocab = self.__load_vocab__()
        self.seg, self.lexicon = self.__load_seg__()
        self.p_model = self.__load_model__()

    def __load_vocab__(self):
        
        with open(os.path.sep.join([cfg.paper.root_path,'weights' ,cfg.paper.vocab]),'rb') as f:
            vocab = pickle.load(f)
        logger.info('加载字典表完成， 字典表长度：%s' % len(vocab))
        print(vocab.stoi)
        return vocab

    # 加载自有字典
    def __load_seg__(self):
        logger.info('加载分词表')
        with open(cfg.paper.lexicon, 'r', encoding='utf-8') as f:
            lexicon = f.readlines()
        lexicon = [x.strip() for x in lexicon]
        seg = pkuseg.pkuseg(user_dict=lexicon)    
        return seg, lexicon

    def __load_model__(self):
        model = TextEmbeddingBagClassify(vocab_size=len(self.vocab), 
                                         embed_dim=cfg.paper.model.embed_dim,
                                         num_class=len(TARGETS))
        model_path = os.path.sep.join([cfg.paper.root_path, 'weights', cfg.paper.model.path])
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        logger.info('加载模型成功，模型参数路径 %s' % model_path)        
        return model

    def __split_words__(self,text):
        text = text.replace('（','(').replace('）',')').replace('．','.').replace('、','.').strip()
        # 后期需做修改，采用正则的方式，防止B. C.这样的也被替换，只替换数字后面的 '.'
        dot_idx = text.find('.')
        if dot_idx < 3:
            text = text[0:dot_idx+1] + re.sub(r'\d+\.',lambda x:x.group().replace('.','-'), text[dot_idx+1:])
        else:
            text = re.sub(r'\d+\.',lambda x:x.group().replace('.','-'), text)


        dot_idx = text.find(')')
        if dot_idx < 4:
            text = text[0:dot_idx+1] + text[dot_idx+1:].replace(')','')
        else:
            text = text.replace(')','')


        dot_idx = text.find('题')
        if dot_idx < 4:
            text = text[0:dot_idx+1] + re.sub(r'\d+题',lambda x:x.group().replace('题','-'), text[dot_idx+1:])
        else:
            text = re.sub(r'\d+题',lambda x:x.group().replace('题','-'), text)





        words = [wd for wd in self.seg.cut(text) if wd in self.lexicon]    
        new_words = []
        for x in words:
            if x not in new_words:
               new_words.append(x)
        if len(new_words) == 0:
            new_words.append('<unk>')               
        return new_words 

    def __words_to_ids__(self,words):

        words_to_ids = [self.vocab.stoi[x] for x in words]
        offsets = torch.tensor([0])
        words_to_ids = torch.tensor(words_to_ids)

        return words_to_ids, offsets




    # 检测文字类别, 标注类型为：( 内容、答案、解析、问题 ) 四大类
    def detect(self, text):
        # text = re.sub('\{img:\d+\}','',text)
        

        if len(text) == 0:
            return TAG_CONTENT,(None, None)

        # 防止 img 出现 1{img:12}. 这种情况
        if text.find('{img') != -1:
            text = re.sub('\{img:\d+\}','',text) + ' {img:1}'  

        # 防止text 后部分出现  一、二、这样类似，引起检测错误
        if len(text) > 5:
            text  = text[0:5] + re.sub('[一|二|三|四|五][、|.|,]','',text[5:])

        
        words =  self.__split_words__(text)
        # print('split words:',words)
        word_ids, word_offsets = self.__words_to_ids__(words)
        # print('word ids:',word_ids, ' offset:', word_offsets)
        label = self.p_model.predict(word_ids, word_offsets)
        # print('predict result:', label.item())
        result = label.item() + 1
        # logger.info('%s {%s} 检测结果 ---> %s' % (text, words, TARGETS[result]))
        qtype = None
        if result == TAG_QUESTION:
            # qno = words[0].replace('(','').replace(')','').replace('.','')
            qno = words[0].replace('(','').replace('.','')
            for key in qn_type_map.keys():
                if qno in qn_type_map[key]:
                    qtype = key
                    break;

        return result, (qtype, words[0])


    # 检测该问题层缺失的问题
    def detect_loss_question(self, q_no_lists, q_level):
        # print('q_level:', q_level)
        q_level_no_lists = qn_type_map[q_level]
        # print('q_level lists :', q_level_no_lists)
        b_idx = q_level_no_lists.index(q_no_lists[0])
        e_idx = q_level_no_lists.index(q_no_lists[-1]) + 1
        loss_lists = [x for x in q_level_no_lists[b_idx:e_idx] if x not in q_no_lists]
        return loss_lists


    def detect_question_type(self, text):
        if text.find('选择题') != -1:
            return Q_TYPE_SELECT
        elif text.find('填空题') != -1:
            return Q_TYPE_EMPTY
        elif text.find('问答题') != -1 or text.find('解答题') != -1:
            return Q_TYPE_QA
        else:
            return Q_TYPE_UNK
        







