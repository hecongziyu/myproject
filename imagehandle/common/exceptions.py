# -*- coding: utf-8 -*-


class ExampleException(Exception):
    def __init__(self, code, msg):
        super(Exception,self).__init__()
        self.code = code
        self.msg = msg

    def __str__(self):
        return 'code:{} message:{}'.format(self.code,self.msg)




