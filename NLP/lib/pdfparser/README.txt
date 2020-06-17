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
    
    context:embeding _ str(current_level_questions)   +   embeding _ str(current_block_contents)
    body:embeding _ char(content)






