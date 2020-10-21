# 分词工具
import pkuseg
pkseg = pkuseg.pkuseg()



# 最简单的分词工作，单词分开
def sim_word_tokenize(text):
    return list(text)


def pkuseg_word_tokenize(text):
    tokens = pkseg.cut(text)
    return tokens






word_tokenize = pkuseg_word_tokenize



