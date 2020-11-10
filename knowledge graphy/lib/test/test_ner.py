from entity.ner.entity_ner import NERecognition
from entity.alignment.entity_align import EntityAlignment
from kg.persist import KnowledgeEntity
from test_crawer import test_baidubaike_parser

# 实体抽取
def test_ner(dict_file, text):
    '''
    实体抽取实现本次采用分词方式，后续可进行修改，修改成深度学习模式
    处理流程：
    1、对text文本进行分词， 取出分词后为n的词组
    2、检测词组是否为所需的词组，（本地查找 --》 baibu baike --》save local file)
    
    '''
    ner = NERecognition(dict_file=dict_file,seg_name='synonyms')
    tokens = ner.get_entity(text)
    
    m_tokens = list(zip(tokens[0], tokens[1]))

    print('tokens:', m_tokens)
    return m_tokens


# 与知识图谱实体进行实体链接、对齐, 注意只对名词
def test_aligment(dict_file, text, entity_file):
    align = EntityAlignment()
    tokens = test_ner(dict_file, text)
    # entity_list = []
    with open(entity_file, 'r', encoding='utf-8') as f:
        entity_list = f.readlines()

    entity_list = [x.split('\t')[0] for x in entity_list]

    n_tokens = [x[0] for x in tokens if x[1] == 'n']

    n_tokens_similars = [(x,align.similar(text1=x, text2_lists=entity_list)) for x in n_tokens]
    return n_tokens_similars


# 测试增加实体
def test_add_entity(tokens):
    '''
    处理流程：
    1、分词、标明词性 。 （N，V)
    2、实体与图谱实体进行链接
    3、处理不能对齐的实体，通过规则匹配的方式， 标明名词的属性。（后期可能通过深度学习直接标明）
    4、增加到知识图谱
    '''
    
    # 1、分词，标明词性， tokens: [('平行', 'n'), ('四边形', 'n'), ('叫做', 'v')]

    # 2、实体与图谱实体进行链接
    kge = KnowledgeEntity()
    kg_entity_lists = kge.get_all_entity()  # 从知识图谱得到所有的实体信息，后期需修改成文件方式
    align = EntityAlignment()  # 实体链接类
    n_tokens = [x[0] for x in tokens if x[1] == 'n']
    tok_link_entity = [(x, align.similar(text1=x,text2_lists=kg_entity_lists)) for x in n_tokens]
    print('tok link entity :', tok_link_entity)


    
    



# 合作关系抽取测试
def test_extract_relation(tokens, e_rel_type='cooperate'):

    # 通过分词后，找到KG中对应的头实体，如尾实体不在KG中，需找到其是否是公司(暂时在实体识别中增加方法通过规则对应)
    # 如存在多个实体，则方式是两个实体 + 中间的 V 进行关系抽取（通过规则匹配的方式)
    pass





if __name__ == '__main__':
    import argparse
    
    args = argparse.ArgumentParser(description="总体分析")
    args.add_argument("--dict_file", help="字典目录",default='D:\\PROJECT_TW\\git\\data\\finance\\dict.txt',type=str, required=False)
    args.add_argument("--data_dir", help="目录",default=r'D:\PROJECT_TW\git\data\kg\crawer',type=str, required=False)
    args = args.parse_args()

    title = '平行四边形'
    text = test_baidubaike_parser(args.data_dir, title)



    # test_ner(args.dict_file, text=text)

    tokens = test_ner(dict_file=args.dict_file, text=text)
    # test_add_entity(tokens)
