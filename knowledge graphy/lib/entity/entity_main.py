# 实体抽取、实体链接、实体关系

class EntityMaster:
    def __init__(self, e_ner, e_link, e_rel, k_graphy):
        '''
        e_ner ：实体识别，抽取
        e_link : 实体链接
        e_rel : 实体关系
        k_graphy : 知识图谱存储

        '''
        self.e_ner = e_ner
        self.e_link = e_link
        self.e_rel = e_rel
        self.k_graphy = k_graphy


    def handle(self, text):
        '''
        实体总体处理流程
        输入：text 文本串
        输出：list[(head entity, rel,  tail entity)] 

        '''

        print('handle text .')

        # STEP 1  抽取实体抽取，顺序返回entity_list  LIST[(entity_name, entity_class)]
        #                                toke_list    LIST((entity_name_1, entity_class), v_name, (entity_name_2. entity_class), ... ...)
        entity_list = self.e_ner.extract_entity(text)


        # STEP 2  实体关系抽取
        entity_rel_list = self.e_rel.extract_rel(entity_list)

        

        # STEP 3  保存知识图谱关系
        # self.k_graphy.save(entity_rel_list)



