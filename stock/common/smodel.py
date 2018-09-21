# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import talib
from scipy import stats
# https://c.m.163.com/news/a/D7SAECKR0511E4EQ.html?spss=newsapp&fromhistory=1  从星际2深度学习环境到神经机器翻译，上手机器学习这些开源项目必不可少
# https://c.m.163.com/news/a/D7VBUQVG0514PFUG.html?spss=newsapp  神经网络：问题与解决方案
# https://c.m.163.com/news/a/D7PTU4I70511AQHO.html?spss=newsapp  斯坦福完全可解释深度神经网络：你需要用决策树搞点事
# https://wenku.baidu.com/view/85518fb8240c844769eaeea5.html t分布 从小样本数据估计总体均值
# p 215

# macd 类指标
# http://mrjbq7.github.io/ta-lib/doc_index.html  采用talib库，文档
class indexModel(object):
    def __init__(self):
        pass

    # 计算未来10天上涨的概率, 参看 p 214
    def probability(self, datas=None):
        # 根据历史记录，计算上涨的总体概率，统计总的上涨天数
        datas = datas.combine_income(ndays=1)
        rp = round(float(len(datas[datas.Flag==2]))/float(len(datas)),4)
        print(rp)
        # 套用计算概率密度公式 ，估计未来10天有6天上涨的趋势： prob = stats.binom.pmf(6,10,p)  （from scipy import stats)
        prob = stats.binom.pmf(6,10, rp)
        return prob

    # 检测两指数的收益的相关性  P224

    # 计算股票的收益置信区间 P230
    def interval(self, datas=None):
        return datas

    # 检测关联股票收益相关性 ，对关键列进行显著性判断 P246
    # 回归模型  P259   p > |t| 错误拒绝原假设的概率

    # P 288 风险评估Var


    # P 303 计算股票投资组合

    # P 358

    # 随机模型
    def random_model(self, datas=None):
        signal = None
        if datas:
            sradom = np.random.choice(a=[-1, 0, 1], size=len(datas), replace=True, p=[0.3, 0.4, 0.3])
            signal = pd.Series(sradom, index=datas.index)
        return signal

    # 移动平均模型 P 363  包含检测模型时间序列是否是平稳，并且不是白噪声，根据检查判断模型是否有效
    def ma(self, datas=None):
        close = datas.Close.values
        real = talib.MA(close,timeperiod=30, matype=0)
        real = pd.DataFrame(data=real,index=datas.index)
        return real

    # K线 https://www.zhihu.com/question/60715922   https://community.bigquant.com/t/%E9%87%8F%E5%8C%96%E5%AD%A6%E5%A0%82-%E7%AD%96%E7%95%A5%E5%BC%80%E5%8F%91%E5%80%9F%E5%8A%A9talib%E4%BD%BF%E7%94%A8%E6%8A%80%E6%9C%AF%E5%88%86%E6%9E%90%E6%8C%87%E6%A0%87%E6%9D%A5%E7%82%92%E8%82%A1/254
    # 习笔记-K线模式识别 https://www.ricequant.com/community/topic/2393 https://www.zhihu.com/collection/144011594
    def kmodel(self, datas=None):
        _open = datas.Open.values
        _high = datas.High.values
        _low = datas.Low.values
        _close = datas.Close.values
        ret = talib.CDLEVENINGDOJISTAR(open=_open,high=_high,low=_low,close=_close)
        ret = pd.DataFrame(data=ret,index=datas.index)
        return ret

    # 根据macdhist直方，当前一天为正，当天为负，则为-1, 反之则为1
    def macd(self, datas=None):
        # df['MACD'],df['MACDsignal'],df['MACDhist'] = talib.MACD(np.array(close),fastperiod=6, slowperiod=12, signalperiod=9)
        pass

class classModel(object):
    def __init__(self):
        pass


    def model_net(self,fields,datas=None):
        # 对需要处理的数据进行归一化处理，防止大数吃掉小数
        # https://www.jianshu.com/p/682c24aef525 用python做数据分析4|pandas库介绍之DataFrame基本操作
        # 归一 https://www.zhihu.com/question/57509028
        # 标准化和归一化什么区别？ https://www.zhihu.com/question/20467170
        # sklearn库中数据预处理函数fit_transform()和transform()的区别 http://blog.csdn.net/quiet_girl/article/details/72517053
        # 需具体了解其实现方式
        from sklearn.preprocessing import MinMaxScaler
        from pybrain.structure import SoftmaxLayer
        from pybrain.datasets import ClassificationDataSet
        from pybrain.tools.shortcuts import buildNetwork
        from pybrain.supervised.trainers import BackpropTrainer
        from pybrain.utilities import percentError
        from pybrain.structure import TanhLayer

        scaler = MinMaxScaler()
        datas[fields] = scaler.fit_transform(datas[fields])

        tran_data = datas[fields].values
        tran_target = datas['Flag'].values
        tran_label = ['Sell','Hold','Buy']

        class_datas = ClassificationDataSet(6, 1, nb_classes=3, class_labels=tran_label)
        print(type(tran_target))
        print(tran_target)
        for i in range(len(tran_data)):
            class_datas.appendLinked(tran_data[i], tran_target[i])

        tstdata_temp, trndata_temp = class_datas.splitWithProportion(0.25)

        print(len(tstdata_temp), len(trndata_temp))

        tstdata = ClassificationDataSet(6, 1, nb_classes=3, class_labels=tran_label)
        trndata = ClassificationDataSet(6, 1, nb_classes=3, class_labels=tran_label)

        for n in range(0, trndata_temp.getLength()):
            trndata.appendLinked(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])

        for n in range(0, tstdata_temp.getLength()):
            tstdata.appendLinked(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])

        tstdata._convertToOneOfMany()
        trndata._convertToOneOfMany()

        tnet = buildNetwork(trndata.indim, 5, trndata.outdim,hiddenclass=TanhLayer,outclass=SoftmaxLayer)
        trainer = BackpropTrainer(tnet,dataset=trndata,batchlearning=True,momentum=0.1,verbose=True,weightdecay=0.01)

        for i in range(5000):
            trainer.trainEpochs(20)
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            testResult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            print("epoch: %4d" % trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % testResult)


        return trainer, class_datas

if __name__ == '__main__':
    from common.sdata import StockData
    sd = StockData(code='600893')
    # sdatas = sd.combine_income(ndays=1)

    im = indexModel()
    ret = im.kmodel(datas=sd.datas)
    # ret = m.ma(datas=sd.datas)
    #prob = m.probability(datas=sd)
    # cmodel = classModel()
    # target = cmodel.model_net(datas=sdatas,fields=(['Open','High','Close','Low','Volume','Amount']))











