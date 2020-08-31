实体抽取、关系抽取

实体： 
1、从内容中取得实体后，可通过 http://zhishi.me/api 接口取得其实体相关信息
2、https://www.nltk.org/_modules/nltk/sentiment/vader.html 情感分析，可参考用于实体、关系抽取  (https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638)
3、https://github.com/huyingxi/Synonyms synonyms可以用于自然语言理解的很多任务：文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等
   synonyms.compare('腾讯', '腾讯公司',seg=False) 实体相似度
   synonyms.seg('dddd') 分词
4、https://openreview.net/pdf?id=ry018WZAZ DEEP ACTIVE LEARNING FOR NAMED ENTITY RECOGNITION  主要看怎么要seq labeling做训练数据
5、https://zhuanlan.zhihu.com/p/79764678 主动学习（Active Learning）-少标签数据学习
6、https://zhuanlan.zhihu.com/p/50184092 NLP中的序列标注问题（隐马尔可夫HMM与条件随机场CRF）！！！
7、https://zhuanlan.zhihu.com/p/78350546 2019年主动学习有哪些进展？答案在这三篇论文里
8、https://openreview.net/pdf?id=ry018WZAZ  ！！！
9、https://zhuanlan.zhihu.com/p/90133637 基于深度学习的NER综述
10、https://zhuanlan.zhihu.com/p/84566617 命名实体识别的一点经验
11、https://www.sciencedirect.com/science/article/abs/pii/S1532046419302047
12、https://www.aitimejournal.com/@akshay.chavan/complete-tutorial-on-named-entity-recognition-ner-using-python-and-keras
13、https://github.com/quincyliang/nlp-public-dataset 中英文实体识别数据集，中英文机器翻译数据集
14、https://github.com/lonePatient/BERT-NER-Pytorch 
15、https://github.com/jidasheng/bi-lstm-crf  ！！！
16、https://pytorch-crf.readthedocs.io/en/stable/ !
17、https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html crf
18、https://zhuanlan.zhihu.com/p/34261803 白话条件随机场
19、https://www.zhihu.com/question/24053383 能否尽量通俗地解释什么叫做熵？
20、https://baijiahao.baidu.com/s?id=1636054958974354930&wfr=spider&for=pc 来认识一下“熵”这个重要的概念
21、https://www.jianshu.com/p/55755fc649b1 如何轻松愉快地理解条件随机场（CRF） !!!!!
22、https://github.com/CLUEbenchmark/CLUENER2020 中文细粒度命名实体识别 Fine Grained Named Entity， 包含数据集
'''
