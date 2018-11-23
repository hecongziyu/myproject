#encoding:utf-8
from torch.utils import data
import numpy as np;
import os
import jieba
import gensim.models.word2vec as w2v

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False
def is_legal(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True
def extract_chinese(line):
    res = ""
    for word in line:
        if is_legal(word):
            res = res + word
    return res;
def words2line(words):
    line = ""
    for word in words:
        line = line + " " + word
    return line


# 数据路径  D:/PROJECT_TW/git/data/nlp/w2v/data/
# 模型路径  D:/PROJECT_TW/git/data/nlp/w2v/
class EduData(data.Dataset):
    # data root 训练数据路径
    # word_vec_path词向量
    def __init__(self, data_root, word_vec_path):
        self.nb_words=40000
        self.max_len=64
        self.word_dim=20  
        
        self.texts,self.labels,self.labels_index,self.index_lables = self.__datahelper__(data_root)
        self.word_vec_path = word_vec_path
        if not os.path.exists('{}{}'.format(word_vec_path,'new_model_big.txt')):
               self.__trainwordvc__()      
        self.word_vec_mod = self.getwc()
        self.len = len(self.labels)
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embedding_matrix = self.__get_embedding_matrix__()
        
    def __getitem__(self, index):
        words = self.texts[index]
#         # 数据补全到64位长度, 补空格
#         words = words.ljst(self.max_len,' ')
#         words = words[0,self.max_len]
        word_ids = [self.word_to_idx[x] for x in words if x in self.word_to_idx]
        return word_ids, self.labels[index], words
    
    def __len__(self):
        return self.len    

    def __trainwordvc__(self):
        big_txt = ''
        for item in self.texts:
            big_txt = '{}\n{}'.format(big_txt,' '.join(item))        
        big_txt = big_txt.encode('utf-8')
        train_word_path = '{}trainword.txt'.format(self.word_vec_path)
        with open(train_word_path,'wb') as f:
            f.write(big_txt)  
        sentences = w2v.LineSentence(train_word_path)
        model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=1)
        model.save('{}new_model_big.txt'.format(self.word_vec_path))
        del big_txt
                   
    def getwc(self):
        model = w2v.Word2Vec.load('{}new_model_big.txt'.format(self.word_vec_path))
        return model;        
    
    def __get_embedding_matrix__(self):
        word_vocb=['']
        for text in self.texts:
            for word in text:
                word_vocb.append(word)
        word_vocb=set(word_vocb)
        vocb_size=len(word_vocb)        
#         texts_with_id=np.zeros([len(self.texts),self.max_len])

        #词表与索引的map
        self.word_to_idx={word:i for i,word in enumerate(word_vocb)}
        self.idx_to_word={self.word_to_idx[word]:word for word in self.word_to_idx}        
        embedding_matrix = np.zeros((self.nb_words, self.word_dim))
        
        for word, i in self.word_to_idx.items():
            if i >= self.nb_words:
                break
            if  self.word_vec_mod.wv.__contains__(word):
                embedding_vector = self.word_vec_mod.wv.get_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector        
        
        return embedding_matrix
                   
    #数据预处理函数，在dir文件夹下每个子文件是一类内容
    def __datahelper__(self,dir):
        #返回为文本，文本对应标签
        labels_index={}
        index_lables={}
        num_recs=0
        fs = os.listdir(dir)
        i = 0;
        for f in fs:
            labels_index[f] = i;
            index_lables[i] = f
            i = i + 1;
        texts = []
        labels = []  # list of label ids
        for la in labels_index.keys():
            print(la + " " + index_lables[labels_index[la]])
            la_dir = dir + "/" + la;
            fs = os.listdir(la_dir)
            for f in fs:
                file = open(la_dir + "/" + f, encoding='utf-8')
                lines = file.readlines();
                text = ''
                for line in lines:
                    if len(line) > 5:
                        line = extract_chinese(line)
                        words = jieba.lcut(line, cut_all=False, HMM=True)
                        text = words
                        # TODO， 需要将数据补全到64位长度，以' '进行填空
                        stext = [' '] * self.max_len
                        if len(text) > self.max_len:
                            stext = text[0:self.max_len]
                        else:
                            stext[0:len(text)]=text
                        
                        texts.append(stext)
                        labels.append(labels_index[la])
                        num_recs = num_recs + 1
        return texts,labels,labels_index,index_lables
    
