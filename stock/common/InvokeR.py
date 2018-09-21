# -*- coding: utf-8 -*-
# 启动RServe http://blog.fens.me/r-rserve-server/
import pyRserve
conn = pyRserve.connect()
conn.eval('''source('test.R')''')
conn.r.testLoadRecord('600016')


