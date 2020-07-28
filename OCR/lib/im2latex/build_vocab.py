from os.path import join
import pickle as pkl
from collections import Counter
import argparse


START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# buid sign2id


class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN, 
                        '0':4, '1':5,'2':6, '3':7, '4':8,'5':9,'6':10,'7':11,'8':12,'9':13}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = len(self.sign2id)

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(data_dir, min_count=5):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    counter = Counter()

    formulas_file =  join(data_dir,'data','im2latex_formulas_custom.txt')
    with open(formulas_file, 'r', encoding='utf-8') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(data_dir,'data','im2latex_train_filter.txt'), 'r') as f:
        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            if idx < len(formulas):
                formula = formulas[idx].split()
                counter.update(formula)

    for word, count in counter.most_common():
        if count >= min_count:
            vocab.add_sign(word)
    
    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file, "len :", len(vocab))
    print('vocab :', vocab.id2sign)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)


def load_vocab(data_dir):
    with open(join(data_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pkl.load(f)
    print("Load vocab including {} words!".format(len(vocab)))

    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building vocab for Im2Latex")
    parser.add_argument("--data_path", type=str,
                        default="D:\\PROJECT_TW\\git\\data\\im2latex", help="The dataset's dir")
    args = parser.parse_args()
    vocab = build_vocab(args.data_path,min_count=1)
