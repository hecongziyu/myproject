import ply.lex as lex
# 为了提高性能，你可能希望使用Python的优化模式（比如，使用-o选项执行Python）。然而，这样的话，Python会忽略文
# 档字串，这是lex.py的特殊问题，可以通过在创建lexer的时候使用optimize选项：
# lexer = lex.lex(optimize=1)
class AnsLexer:
    tokens = [
       'SegAnswer',            # 基于数字的题号  1. 1) (1)
       'SegKnowledge',        # 特殊数据题号   1--4
       'SegResolve',
       'SegContent',
     ] 
    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'

#     t_SegAnswer = r'(故答|故选|答案[:|.|：])'
#     t_SegKnowledge = r'(考点|知识点)[:|.|：]'
#     t_SegResolve = r'(解析|解答)[:|.|：]'
#     t_SegContent = r'[\s\S]'
    
    def __init__(self):
        self.ans_map = {'Answer': '', 'Knowledge':'','Resolve':'' }
        self.curr_ans_key = 'Answer'
        
    def t_SegKnowledge(self, t):
        r'(考点|知识点)[:|.|：]'
        self.curr_ans_key = 'Knowledge'
#         self.ans_map[self.curr_ans_key] = '{}{}'.format(t.value, self.ans_map[self.curr_ans_key])
        return t
    
    def t_SegAnswer(self,t):
        r'(故答|故选|答案[:|.|：])'
        self.curr_ans_key = 'Answer'
#         self.ans_map[self.curr_ans_key] = '{}{}'.format(t.value, self.ans_map[self.curr_ans_key])
        return t
    
    def t_SegResolve(self,t):
        r'(解析|解答)[:|.|：]'
        self.curr_ans_key = 'Resolve'
#         self.ans_map[self.curr_ans_key] = '{}{}'.format(t.value, self.ans_map[self.curr_ans_key])
        return t
        
    def t_SegContent(self, t):
        r'[\s\S]'
        self.ans_map[self.curr_ans_key] = '{}{}'.format(self.ans_map[self.curr_ans_key], t.value)
        return t

    # Define a rule so we can track line numbers
#     def t_newline(self,t):
#         r'\n+'
#         t.lexer.lineno += len(t.value)
    # Error handling rule
    def t_error(self,t):
        # print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)
    # Build the lexer
    def build(self,**kwargs):
        self.lexer = lex.lex(module=self, **kwargs)
        
    def input(self, input):
        self.lexer.input(input)
        tok_lists = []
        while True:
            tok = self.lexer.token()
            if not tok: break;
