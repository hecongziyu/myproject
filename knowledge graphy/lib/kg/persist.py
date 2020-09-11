from py2neo import Graph




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
        q_result = self.graph.run(query).data()
        if len(q_result) == 0:
            return None
        result = [x['n'].get('name') for x in q_result]
        return result





