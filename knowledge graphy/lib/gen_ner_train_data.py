from os.path import join
from entity.ner import NERecognition
# 生成命名实体、关系抽取训练数据

def gen_ner_train_data(data_root, seed_file):
    '''
    生成命名实体训练文件
    处理流程：
    1、
    '''
    
    # 2、通过种子文件调用爬虫
    pass


# 生成关系抽取训练数据处理， 对爬虫文件、种子文件进行规则化、分词词典
def gen_nre_train_data(data_root, crawer_file, seed_file, re_gen_file=False):
    if re_gen_file:
        # 1、将种子文件转化成分词字典
        with open(join(data_root,'nre',seed_file), 'r', encoding='utf-8') as f:
            contents = f.readlines()
        contents = [x.strip() for x in contents]
        contents = ['{}\tnkn'.format(x) for x in contents]
        with open(join(data_root, 'nre', 'dict_file.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(contents))

        # 2、读取文件，将文件按句号隔开并保存
        with open(join(data_root,'nre', crawer_file), 'r', encoding='utf-8') as f:
            contents = f.readlines()
        contents = [x.strip() for x in contents]
        contents = ''.join(contents)
        contents = contents.split('。')
        contents = [x+'.' for x in contents]
        with open(join(data_root,'nre', 'pre_sentence_file.txt'),'w', encoding='utf-8') as f:
            f.write('\n'.join(contents))

    # 3、生成NRE训练数据，注意本次训练采用snow ball 半监督模式进行训练，需要对
    #    sendtence file 中的数学知识点进行标签
    with open(join(data_root,'nre', 'pre_sentence_file.txt'),'r', encoding='utf-8') as f:
        contents = f.readlines()

    ner = NERecognition(dict_file=join(data_root, 'nre', 'dict_file.txt'),seg_name='pkuseg')
    content_tag_lists = []
    for item in contents:
        entity_lists = ner.extract_entity(item)
        # print('text :', item , ' entity_lists:', entity_lists)
        _tmp_str = ''
        for item in entity_lists:
            _str,  _pos = item
            if _pos == 'nkn':
                _tmp_str = '{}<NKN>{}</NKN>'.format(_tmp_str, _str)
            else:
                _tmp_str = '{}{}'.format(_tmp_str, _str)

        content_tag_lists.append(_tmp_str)
    
    with open(join(data_root,'nre', 'sentence_file.txt'),'w', encoding='utf-8') as f:
        f.write('\n'.join(content_tag_lists))

    print('训练数据生成完成.')
    




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="试卷导入功能")
    parser.add_argument("--data_root", default=r"D:\PROJECT_TW\git\data\kg\entity", help="模型参数、分词", type=str)
    parser.add_argument("--crawer_file", default=r"crawer_file.txt", help="模型参数、分词", type=str)    
    parser.add_argument("--seed_file", default=r"seed_file.txt", help="模型参数、分词", type=str)    
    args = parser.parse_args()
    gen_nre_train_data(data_root=args.data_root, crawer_file=args.crawer_file, seed_file=args.seed_file, re_gen_file=True)



