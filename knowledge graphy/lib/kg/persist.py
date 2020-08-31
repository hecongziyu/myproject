from py2neo import Graph
from .init import cfg




# 知识点实体
class KnowledgeEntity:
    def __init__(self, url='bolt://127.0.0.1:7687', auth=('neo4j','654321')):
        self.url = url
        self.auth = auth
        self.graph = Graph(uri=self.url,auth=self.auth)

    def add_entity(self, entity_name):
        pass


    def add_entity_rel(self, head_entity, tail_entity, rel_name):
        pass


    def get_all_entity(self):
        query = 'MATCH (n:Knowledge) RETURN n'
        result = graph.run(query).data()
        if len(result) == 0:
            return None
        return result[0]['n'].get('name')





