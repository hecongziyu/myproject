# -*- coding: UTF-8 -*-
# https://www.jianshu.com/p/0eaeba15ee68 
import ply.lex as lex

# List of token names.   This is always required


def lexer():
    tokens = (
       # 'SegNumOne',            # 基于数字的题号  1. 1) (1)
       'SegNum',            # 基于数字的题号  1. 1) (1)
       'SegNumSpec',        # 特殊数据题号   1--4

    )    
    # t_SegNumOne   = r'[\(|（][1-9][0-9]{0,1}[\)|）]'
    t_SegNum   = r'[\(|（]{0,1}[1-9][0-9]{0,1}[）|)]{0,1}[\.|．|\s|\t]{1}(?!\d+)'
    t_SegNumSpec = r'[1-9][0-9]{0,1}[-|－|—]{1,4}[1-9][0-9]{0,1}[\s]{0,5}[A-E]{2,5}'

    # Define a rule so we can track line numbers
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'

    # Error handling rule
    def t_error(t):
        # print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer
    return lex.lex()



# 问题级别解析
# def lexer_level():
#     tokens = (
#        'LevelNormOne',            # 基于数字的题号  1. 2. 3. 4.
#        'LevelNormTwo',            # 基于数字的题号  1) 2) 3)
#        'LevelNormThree',          # 特殊数据题号   (1)  (2)  (3)
#        'LevelAlphaOne',           # （一），（二），（三）

#     )  