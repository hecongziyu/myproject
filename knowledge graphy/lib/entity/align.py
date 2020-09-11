import difflib
'''
实体对齐
1、https://www.jianshu.com/p/853d86e090a7 python自带比较相似度的模块，difflib
'''

class AlignmentSimple:
    '''
    python自带比较相似度的模块，difflib
    '''
    def __init__(self):
        pass

    def __call__(self, text1, text2):
        # lambda x:x=" "， None 不想算在内的元素， 在isjunk是否跳过的值
        text1 = text1.replace('公司','').replace('有限','').replace('无限','').replace('责任','').replace('信息','').replace('科技','')
        text2 = text2.replace('公司','').replace('有限','').replace('无限','').replace('责任','').replace('信息','').replace('科技','')
        return round(difflib.SequenceMatcher(None, text1, text2).quick_ratio(),3)



class EntityAlignment:
    '''
    实体对齐， 实体链接，暂时用较简单方式，python 自带的difflib
    后期采用深度学习等基它方式进行对齐，包括属性等等
    '''
    def __init__(self, align_type='simple', sim_threshold=0.8):
        self.alignment = None
        if align_type == 'simple':
            self.alignment = AlignmentSimple()


    def entity_similar(self, text1, text2=None, text2_lists=None, sim_threshold=0.8):
        if text2 is not None:
            return self.alignment(text1, text2)
        elif text2_lists is not None:
            sim_lists =  [(x, self.alignment(text1,x)) for x in text2_lists]
            sim_lists.sort(key=lambda x:x[1], reverse=True)
            sim_lists =  [x[0] for x in sim_lists if x[1] >= sim_threshold]
            return sim_lists



