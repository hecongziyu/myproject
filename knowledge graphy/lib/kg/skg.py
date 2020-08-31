import sys
sys.path.append('D:\\PROJECT_TW\\git\\finance')    
import lib.kg.external as gather
import lib.kg.persist as dao


# 知识图谱

# 关系
Relations = ['Manager','BelongTo','Compete','Cooperate']

# 关系、实体类型对应关系
Entity_Relations_Define = {
    'BelongTo':('Stock', 'Concept')
}

class SKnowledgeGraphy:
    def __init__(self):
        pass

    def add_rel(self, ):
        raise Exception('add knowledge graphy relation not implement ！')


# 规则数据的实体关系抽取
class SRuleKnowledgeGraphy(SKnowledgeGraphy):
    def __init__(self):
        SKnowledgeGraphy.__init__(self)

    # 增加KG实体关系
    def add_rel(self, code, rel_name):
        s_name = dao.get_name_by_code(code)
        rel_gather = gather.KGRuleDataSinaGather(rel_name)
        rel_name, rel_entitys = rel_gather(code=code)
        entity_rel_lists = [(s_name, rel_name, x) for x in rel_entitys]
        print('entity_rel_lists', entity_rel_lists)
        dao.add_entity_relation(entity_rel_lists, Entity_Relations_Define[rel_name])


    


if __name__ == '__main__':
    skg = SRuleKnowledgeGraphy()
    skg.add_rel('002261','BelongTo')
    # get_entity_by_name('拓维信息')




    



