import ply.lex as lex
'''
https://www.jianshu.com/p/0eaeba15ee68 语法分析

'''


class EntityRelation:
    '''
        实体关系抽取
    '''    
    def __init__(self):
        pass


    def __entity_rel_group__(token_lists):
        
        pass

    def extract_rel(self,token_lists):
        '''
        通过正则或规则匹配方式取得关系，后期修改为深度学习模式
        处理流程：
        1、分割 token lists 分成 list[(n,v,n),(n,v,n)]
        2、正则方式得到关系
        '''
        print('extract relation ')