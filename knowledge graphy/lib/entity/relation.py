import ply.lex as lex
'''
https://www.jianshu.com/p/0eaeba15ee68 语法分析
https://zhuanlan.zhihu.com/p/143867739 Lex与YACC详解
https://www.cnblogs.com/hdk1993/p/4922801.html

实体抽取、关系抽取
关系： 合作（公司名， 属性：合作行业）   ：cooperate
       


关系抽取
1、https://tech.paayi.com/nlp-relation-extraction-in-python
2、https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/  ！！！！
3、https://baijiahao.baidu.com/s?id=1650526419042842719&wfr=spider&for=pc 实体关系抽取的现状与未来
4、https://www.sohu.com/a/303259496_312708 史上最大实体关系抽取数据集 | 清华发布
5、https://zhuanlan.zhihu.com/p/52722394 《FewRel》阅读笔记
6、https://zhuanlan.zhihu.com/p/44772023 知识抽取-实体及关系抽取
7、https://www.jianshu.com/p/9d33520f2a68 利用关系抽取构建知识图谱的一次尝试
8、https://www.cnblogs.com/vpegasus/p/re.html 自然语言处理(一) 关系抽取 !!!!
9、https://www.cnblogs.com/theodoric008/p/7874373.html 实体关系抽取 entity relation extraction 文献阅读总结
10、https://zhuanlan.zhihu.com/p/44772023 实体、关系抽取
11、https://zhuanlan.zhihu.com/p/69362160 医疗健康文本的关系抽取和属性抽取
12、https://zhuanlan.zhihu.com/p/138127188 实体链接 任务介绍
13、https://zhuanlan.zhihu.com/p/122386084 Ranking loss系列 孪生神经网络(Siamese Network)
14、https://www.zhihu.com/question/264595128 什么是meta-learning?
15、http://nlpprogress.com/english/relationship_extraction.html  包含模型和代码
16、https://github.com/didi/ChineseNLP  ！！！！包含数据集
17、https://www.kesci.com/home/dataset/5dde487dca27f8002c4a8352 关系抽取数据集
18、https://www.52nlp.cn/tag/%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96-relation-extraction ！！！！
19、https://www.cnblogs.com/veagau/p/11748248.html 关于N-Way K-Shot 分类问题的理解 ！！！！
20、https://www.zhihu.com/question/363200569/answer/956974860  N-Way K-Shot 分类 有开源的代码


1、http://www.doc88.com/p-1915064602123.html 基于强化学习的关系抽取系统设计与实现  ！！！
2、https://www.pianshen.com/article/6907257201/ 关系抽取常用方法
3、https://www.ixueshu.com/document/2effc9790427aa6d318947a18e7f9386.html 基于词形规则模板的术语层次关系抽取方法
4、http://www.mathcs.emory.edu/~eugene/papers/dl00.pdf Snowball:ExtractingRelationsfromLarge Plain-TextCollections
5、http://www.guayunfan.com/baike/70592.html 概念关系抽取的方法_多语种叙词本体
6、https://nlp.stanford.edu/pubs/chambers-acl2011-muctemplates.pdf
7、https://github.com/thunlp/OpenNRE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
8、https://www.jianshu.com/p/99cbfc1779c6 基于依存分析的开放式中文实体关系抽取方法 ！！！
9、https://zhuanlan.zhihu.com/p/44772023 ！！

11、https://www.ixueshu.com/document/a8b9855222fc772c54f40040fa596289318947a18e7f9386.html ！
12、https://blog.csdn.net/u012736685/article/details/97616693 知识图谱（六）——关系抽取
13、https://github.com/thunlp/OpenNRE
14、https://github.com/thunlp/NREPapers ！！！
15、https://blog.csdn.net/li15006474642/article/details/104683237 ！！！！
16、https://blog.csdn.net/bbbeoy/article/details/80885753 Bootstrap方法详解——技术与实例 ！！！
17、https://blog.csdn.net/SunJW_2017/article/details/79160369 统计学中的Bootstrap方法介绍及其应用
18、https://www.jianshu.com/p/2eafca726178 Bootstraping
19、https://blog.csdn.net/parasol5/article/details/49050537 关于文本挖掘系统snowball
20、https://xueshu.baidu.com/usercenter/paper/show?paperid=becca2b57bedf3de2f8cb4cf6faa2737&site=xueshu_se
21、https://zhuanlan.zhihu.com/p/101058270 关系抽取：Snowball System ！！！！！
22、https://blog.csdn.net/weixin_46249816/article/details/105754292 笔记:关系抽取算法之Snowball   ！！！！！
知识图谱之关系抽取代码实战 23、https://blog.csdn.net/qq_40176087/article/details/103975642?utm_medium=distribute.pc_relevant.none-task-blog-title-6&spm=1001.2101.3001.4242
斯坦福大学-自然语言处理入门 笔记 第十课 关系抽取 24、https://blog.csdn.net/kunpen8944/article/details/83182845?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242
25、http://www.cs.columbia.edu/~gravano/Papers/2000/dl00.pdf  Snowball.pdf
26、http://www.doc88.com/p-908997185266.html 自动关系抽取的三种重要方法 共75页
27、https://zhuanlan.zhihu.com/p/91007237 一种简单的文本聚类算法：single-pass
https://www.ijcit.com/archives/volume5/issue5/Paper050503.pdf
https://github.com/jalajthanaki/NLPython
https://github.com/shibukawa/snowball_py ！！！
http://www.ambuehler.ethz.ch/CDstore/www2009/proc/docs/p101.pdf  StatSnowball 使用判别式Markov logic networks (MLNs)，并通过极大似然估计来学习网络的权值
！！！！ https://github.com/thunlp/Neural-Snowball#:~:text=%20README.md%20%201%20Step%201%20Pre-train%20the,3%20Step%203%20Start%20neural%20snowball%20More%20 
https://cn.bing.com/search?q=git+relation+extraction+nlp+base+on+pattern&qs=n&form=QBRE&sp=-1&pq=git+relation+extraction+nlp+base+on+pattern&sc=0-43&sk=&cvid=8E2789F06C9C4EB4B81E947ED2DE2334
git relation extraction nlp base on pattern
！！！！https://github.com/thunlp/Neural-Snowball#:~:text=%20README.md%20%201%20Step%201%20Pre-train%20the,3%20Step%203%20Start%20neural%20snowball%20More%20
https://github.com/shibukawa/snowball_py
https://github.com/davidsbatista/Snowball ！！！
https://github.com/davidsbatista/BREDS ！！！
https://www.aclweb.org/anthology/D15-1056.pdf ！！！


https://medium.com/towards-artificial-intelligence/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0 ！！
'''


class EntityRelation:
    '''
        实体关系抽取
    '''    
    def __init__(self):
        pass


    def __entity_rel_group__(token_lists):
        
        pass

    def extract_rel(self,token_lists):
        '''
        通过正则或规则匹配方式取得关系，后期修改为深度学习模式
        处理流程：
        1、分割 token lists 分成 list[(n,v,n),(n,v,n)]
        2、正则方式得到关系
        '''
        print('extract relation ')