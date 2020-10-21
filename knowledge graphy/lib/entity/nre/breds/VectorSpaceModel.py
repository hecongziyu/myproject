#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.cnblogs.com/lesliechan/p/11966642.html 基于word2vec 词向量训练
# https://zhuanlan.zhihu.com/p/111754138 自然语言处理——使用词向量（腾讯词向量）
# 中文词向量 https://mlln.cn/2018/06/28/%E6%9C%80%E5%85%A8%E4%B8%AD%E6%96%87%E8%AF%8D%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E4%B8%8B%E8%BD%BD-%E9%83%BD%E6%98%AF%E8%AE%AD%E7%BB%83%E5%A5%BD%E7%9A%84%E4%BC%98%E8%B4%A8%E5%90%91%E9%87%8F/  
# https://www.jianshu.com/p/091383e86825 Tf-Idf详解及应用

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
import codecs
import re

from gensim import corpora
from .tokens import word_tokenize
from gensim.models import TfidfModel


class VectorSpaceModel(object):

    def __init__(self, sentences_file, stopwords):
        self.dictionary = None
        self.corpus = None
        f_sentences = codecs.open(sentences_file, encoding='utf-8')
        documents = list()
        count = 0
        print("Gathering sentences and removing stopwords")
        for line in f_sentences:
            line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)

            # remove stop words and tokenize
            document = [word for word in word_tokenize(line.lower()) if word not in stopwords]
            documents.append(document)
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")
        f_sentences.close()


        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in documents]
        self.tf_idf_model = TfidfModel(self.corpus)

        print(len(documents), "documents read")
        print(len(self.dictionary), " unique tokens")
