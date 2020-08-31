'''
知识图谱
一、基础
1、实体: 人名、公司、行业、
2、关系：高管、竞争、合作、属于
3、属性：如：行业占公司比例 

二、数据来源
1、规则数据 (sina api)

主要问题：不通用，需要对图谱关系进行单独编写接口
计划：先做行业的， 可用于新闻的情感分析

2、不规则数据， 新闻、微博
处理流程：
1） 实体识别 --> 包括 实体类型（公司名、行业名等）
2） 实体间关系识别 -->  
    a. 先利用分词得到实体及其属性(后期通过模型训练得到)， 动词
    b、暂时通过动词匹配的方式得到其关系。

'''

