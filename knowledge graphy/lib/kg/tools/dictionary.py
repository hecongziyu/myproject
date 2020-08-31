# 字典管理，在字典中新增词汇信息
# 需维护的字典信息：情感、公司、产品信息、行业信息

'''
1、 https://blog.csdn.net/weixin_40411446/article/details/81014669 金融领域词典构建
2、https://www.cnblogs.com/en-heng/p/5848553.html TF-IDF提取行业关键词
3、https://blog.csdn.net/qq_40676033/article/details/88038212 中文情感词典的构建
4、https://github.com/huyingxi/Synonyms synonyms可以用于自然语言理解的很多任务：文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等
   synonyms.compare('腾讯', '腾讯公司',seg=False) 实体相似度
   synonyms.seg('dddd') 分词
'''

'''
流程：
1、设置种子查询实体， 调用http://zhishi.me/api 接口取得其实体相关信息
2、通过取得的实体信息，再去取其它实体信息
3、重构实体字典
'''


