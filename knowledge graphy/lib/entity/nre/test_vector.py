# 测试词向量  tf_idf model  后期改成其它词向量模式
# TfidfModel 无需训练

from breds.VectorSpaceModel import VectorSpaceModel
from os.path import join
from breds.tokens import word_tokenize
import pickle
from gensim.matutils import cossim
from breds.sentence import Sentence
from breds.config import Config
from breds.tuples import Tuple
from breds.pattern import Pattern

# 
def init_vector_space(sentence_file, stop_words):
    vsm = VectorSpaceModel(sentence_file, stop_words)
    f = open("vsm.pkl", "wb")
    pickle.dump(vsm, f)
    f.close()


# 检测两段话相似度
def test_vector_similarity(config, text1, text2, stop_words):
    with open("vsm.pkl", "rb") as f:
        vsm = pickle.load(f)

    print('text 1 token :', [x for x in word_tokenize(text1) if x not in stop_words])
    print('text 2 token :', [x for x in word_tokenize(text2) if x not in stop_words])

    vector_ids_1 = vsm.dictionary.doc2bow([x for x in word_tokenize(text1) if x not in stop_words])
    vector_ids_2 = vsm.dictionary.doc2bow([x for x in word_tokenize(text2) if x not in stop_words])

    print('text 1:', text1, '\nvector ids 1:', vector_ids_1, '\n vector 1:', config.vsm.tf_idf_model[vector_ids_1])
    print('text 2:', text2, '\nvector ids 2:', vector_ids_2, '\n vector 2:', config.vsm.tf_idf_model[vector_ids_2])

    sim = cossim(vector_ids_1, vector_ids_2)

    print('similarity:', sim)


# 注意测试pattern vector采用tf-idf模型， 在parameters 配置文件中 use_reverb 设为 no (不采用词性检测)
def test_pattern_similarity(config, text1, text2, stop_words):
    # t = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after,self.config)    
    # pattern1 = Pattern(matched_tuples[0])
    # pattern2 = Pattern(matched_tuples[0])
    pattern = None
    sentence1 = Sentence(text1, 
                        config.e1_type, 
                        config.e2_type, 
                        config.max_tokens_away,
                        config.min_tokens_away, 
                        config.context_window_size)   
    for rel in sentence1.relationships: 
        # rel_1 = sentence1.relationships
    # t_1 = Tuple(rel_1.ent1, rel_1.ent2, rel_1.sentence, rel_1.before, rel_1.between, rel_1.after,config)    
        print('before:', rel.before, ' between:', rel.between)
        tup = Tuple(rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after,config)    
        print('before vector:', tup.bef_vector)
        print('between vector:', tup.bet_vector)
        if pattern is None:
            pattern = Pattern(tup)
        else:
            pattern.add_tuple(tup)


    print('len pattern tuples:', len(pattern.tuples))
    print('pattern before vector:', pattern.centroid_bef)
    print('pattern between vector:', pattern.centroid_bet)
    




if __name__ == '__main__':
    from os.path import join

    # init_vector_space(join(data_root, sentence_file), ['.','。',','])



    
    data_root = r'D:\PROJECT_TW\git\data\kg\entity\nre'
    configuration = 'parameters.cfg'
    sentences_file = 'sentence_file.txt'
    seeds_file = 'seeds_positive.txt'
    negative_seeds = 'seeds_negative.txt'
    similarity = 0.5
    confidance = 0.5

    config = Config(configuration, join(data_root,seeds_file), 
                                 join(data_root, negative_seeds),
                                 join(data_root, sentences_file),
                                 similarity, 
                                 confidance)

    

    text1 = '两腰相等的<NKN>梯形</NKN>叫做<NKN>等腰梯形</NKN>,两腰相等的<NKN>梯形</NKN>叫做<NKN>等腰梯形</NKN>'
    text2 = '两组对边分别平行的,平行'
    # test_vector_similarity(config, '分别', '两组分别', ['.','。',','])
    test_pattern_similarity(config, text1, text2, ['.','。',','])