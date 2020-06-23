一、说明：
    解析试卷WORD，自动分题


二、试卷内容类型
    标题、分级问题（一级、二级、三级、四级）, 内容

    解析PDF后包含信息：
    1）文字内容、文字位置信息


三、分题思路

    def check_content(title, current_level_questions, current_block_contents, content):
        '''
        title: 标题
        current_level_questions   当前问题列表, [问题级别、分词、位置信息]
        current_block_contents  当前问题区域内容列表
        content: 需判断的内容

        返回结果：标题， 第几级问题、普通内容
        '''
        pass


四、模型：
1 )
    CRN :
        x1 = encoding(str)
        c =  list str
        P(xn|cn) 

2 )     
    E = TimeSeries(e1, e2, e3, e4, ... en)

    e1 = str , position_type(标题、问题分级、内容)

    str encoding = lstm(embeding(max len 10 char) ) 
    context encoding = vstack(str encoding list)


 
    ? 

    context:embeding _ str(current_level_questions)   +   embeding _ str(current_block_contents)
    body:embeding _ char(content)


3）LOSS
    
    input :  list[str]  target:   list[label]

    loss = crossentry(preditct_target, labels)



五、训练数据
试卷ID | 试题内容  | 类型 , ( 问题 ) 级别
1| str | type and level





 