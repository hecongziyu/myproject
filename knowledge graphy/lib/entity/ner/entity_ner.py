import synonyms
import pkuseg
'''
实体识别-
'''
class NERSynonyms:
    '''
    基于分词器进行实体识别
    include_token指定 字性类型
    '''
    def __init__(self,dict_file,include_token=['n','v']):
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
        return tokens

    def __adjust_tokens__(self, tokens,token_pos_list):
        pass




class NERecognition:
    '''
    NER识别，分为两种方式
    1、规则匹配
    2、深度学习（等规则匹配完成后，通过程序生成训练数据再做）
    '''
    def __init__(self,dict_file,seg_name='synonyms'):
        self.dict_file = dict_file
        self.rec_map = {
            'synonyms':NERSynonyms(dict_file=self.dict_file),
            'pkuseg':NERPkuseg(dict_file=self.dict_file)
        }
        self.seg = self.rec_map[seg_name]

    # 得到实体，词性
    def get_entity(self, text):
        return self.seg(text)



