# -*- coding: UTF-8 -*-
from collections import Counter
import jieba
from itertools import product

# UNK_TOKEN = 0

# class WordConvert:
#    def __init__(self):
#        #initialize containers to hold the words and corresponding index
#        self.word2index = {"UNK": UNK_TOKEN}
#        self.index2word = {UNK_TOKEN: "UNK"}
#        self.n_words = 1  # Count SOS and EOS

#    def addSentence(self, sentence):
#        for word in sentence.split(' '):
#            self.addWord(word)

#    def addWord(self, word):
#        if word not in self.word2index:
#            self.word2index[word] = self.n_words
#            self.index2word[self.n_words] = word
#            self.n_words += 1

#     def __len__(self):
#         return self.n_words


# 生成题号
def gen_question_no():
    question_no= [list(range(1,99)), 
                        '一,二,三,四,五,六,七,八,九,十'.split(','),
                        'A,B,C,D,E,F,G'.split(',')]
    # 不需加标点符号
    querstion_no_special = '①,②,③,④,⑤,⑥,⑦,⑧,⑨,⑩'.split(',')

    # 标点符号
    punctuation = ['.',',','、', ')' ,'）', ('(',')'), ('（', '）')]

    qn_lists = []

    for qls in question_no:
        for qitem, pitem in product(qls, punctuation):
            if type(pitem) == tuple:
                q_n = '{}{}{}'.format(pitem[0], qitem, pitem[1])
            else:
                q_n = f'{qitem}{pitem}'
            qn_lists.append(q_n)
    qn_lists.extend(querstion_no_special)
    # print(len(qn_lists))
    return qn_lists


# 创建字典
def build_vocab(data_dir, min_count=3):
    # min count 文字最小在文档中出现的次数
    word_convert = WordConvert()
    counter = Counter()

    data_file = os.path.sep.join([data_dir, 'all_file.txt'])

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = jieba.lcut(line, cut_all=False, HMM=True)
            counter.update(words)

    for word, count in counter.most_common():
        if count > min_count:
            word_convert.addWord(word)

    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file, "len :", len(vocab))
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)


def load_vocab(data_dir):
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))
    return vocab


if __name__ == '__main__':
    pass