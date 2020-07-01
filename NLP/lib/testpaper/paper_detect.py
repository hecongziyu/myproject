# -*- coding: UTF-8 -*-
from model import *
from txtdataset import TARGETS,STOP_WORDS,TAG_QUESTION,TAG_CONTENT,TAG_ANSWER,TAG_ANSWER_AREA
from config import _C as cfg, logger
import pkuseg
import numpy as np
from utils.txt_utils import gen_question_no,gen_question_no_type

# 题号信息
qn_lists = gen_question_no()
qn_type_map = gen_question_no_type()

class PaperDetect(object):
    def __init__(self):
        # self.use_cuda = True if cfg.model.cuda and torch.cuda.is_available() else False
        # self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.vocab = self.__load_vocab__()
        self.seg, self.lexicon = self.__load_seg__()
        self.p_model = self.__load_model__()

    def __load_vocab__(self):
        return None

    # 加载自有字典
    def __load_seg__(self):
        with open(cfg.paper.lexicon, 'r', encoding='utf-8') as f:
            lexicon = f.readlines()
        lexicon = [x.strip() for x in lexicon]
        seg = pkuseg.pkuseg(user_dict=lexicon)    
        return seg, lexicon

    def __load_model__(self):
        # pmodel = TextClassify(vocab_size=len(self.vocab), embed_dim=cfg.model.embed_dim, 
        #                         hidden_dim=cfg.model.hidden_dim, num_class=len(TARGETS))
        # pmodel.load_state_dict(torch.load(cfg.model.path))
        # pmodel.to(self.device)
        # pmodel.eval()
        # return pmodel
        return None



    # 检测文字类别, 标注类型为：( 内容、答案、解析、问题 ) 四大类
    def detect(self, text):
        text = text.replace('（','(').replace('）',')').replace('．','.').replace('、','.').replace(',','.').replace('，','.').strip()
        words =  self.seg.cut(text)
        if words[0] in qn_lists:
            qno = words[0].replace('(','').replace(')','').replace('.','')
            qtype = None
            for key in qn_type_map.keys():
                if qno in qn_type_map[key]:
                    qtype = key
                    break;
            return TAG_QUESTION, qtype
        else:
            return TAG_CONTENT,None






