# 测试模块分词、相似度
# import sys
# sys.path.append('./breds')
from os.path import join
from breds.sentence import Sentence
from breds.config import Config
import codecs
from breds.tokens import word_tokenize


def test_sentence(config, s_file):
    sentences_lists = codecs.open(s_file, encoding='utf-8')
    i = 0
    for line in sentences_lists:
        print('line -->', line)
        sentence = Sentence(line.strip(), 
                            config.e1_type, 
                            config.e2_type, 
                            config.max_tokens_away,
                            config.min_tokens_away, 
                            config.context_window_size)

        print('rel length :', len(sentence.relationships))
        for rel in sentence.relationships:
            if rel.arg1type == config.e1_type and rel.arg2type == config.e2_type:
                print('sentence :', rel.sentence)
                print('ent 1:', rel.ent1, 'ent 2:', rel.ent2)
                print('rel between :', rel.between)
                print('rel before :', rel.before)
                print('tokens :', word_tokenize(rel.between))
        if i == 0:
            break
        i += 1


if __name__ == '__main__':
    # if len(sys.argv) != 7:
    #     print("\nBREDS.py parameters sentences positive_seeds negative_seeds "
    #           "similarity confidence\n")
    #     sys.exit(0)
    # else:
    from os.path import join
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

    test_sentence(config, join(data_root, sentences_file))




