# -*- coding: utf-8 -*-
import talib
import numpy as np
import pandas as pd


# 相关参考资料
# 1、https://zhuanlan.zhihu.com/p/25407061?refer=uqer2015 TaLib在股票技术分析中的应用
# 2、http://bbs.pinggu.org/thread-5720092-1-1.html 借助talib使用技术分析指标来炒股（附完整策略代码）
class IndexModel(object):
    def __init__(self):
        pass

    # https://zhuanlan.zhihu.com/p/27491500
    # MACD是从双指数移动平均线发展而来的，由快的指数移动平均线（EMA12）减去慢的指数移动平均线（EMA26）得到快线DIF，再用2×（快线DIF-DEA）得到MACD柱。MACD的意义和双移动平均线相似，即由快、慢均线的离散、聚合来显示当前的多空状态和股价可能的发展变化趋势并对买进、卖出时机作出研判，但MACD阅读起来更方便

    def macd(self, datas, fastperiod=12, slowperiod=26, signalperiod=9):
        # 初始化交易信号
        tsignal = pd.Series(np.ones((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        # 得到三个时间序列数组，分别为macd, signal 和 hist
        price = datas.Close.values
        macd, signal, hist = talib.MACD(price, fastperiod, slowperiod, signalperiod)
        for idx in range(40, len(datas)):
            if macd[idx ] - signal[idx ] < 0 and macd[idx - 1] - signal[idx - 1] > 0:
                # 卖出逻辑 macd下穿signal， 卖出信号
                tsignal[idx] = 0
            if macd[idx ] - signal[idx] > 0 and macd[idx - 1] - signal[idx - 1] < 0:
                # 买入逻辑  macd上穿signal, 买入信号
                tsignal[idx] = 2
        return tsignal,macd, signal, hist

    # 布林带指标
    # 布林线（Bollinger Band）是由三条线组成，在中间的通常为 20 天平均线，而在上下的两条线则分别为 Up 线和 Down 线
    # 策略实现
    # 1、股价高于这个波动区间，即突破上顶，说明股价虚高，故卖出
    # 2、股价低于这个波动区间，即穿破下底，说明股价虚低，故买入
    def bband(self, datas, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        # 初始化交易信号
        tsignal = pd.Series(np.zeros((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        close = datas.Close.values
        upperband, middleband, lowerband = talib.BBANDS(close,timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

        for idx in range(30, len(datas)):
            if datas.High.values[idx] > upperband[idx]:
                tsignal[idx] = -1

            if datas.Low.values[idx ] < lowerband[idx]:
                tsignal[idx] = 1

        return tsignal,upperband, middleband, lowerband

    # 资金流量指标（MFI，英文全名Money Flow Index）是相对强弱指标（RSI）和人气指标（OBV）两者的结合。MFI指标可以用于测度交易量的动量和投资兴趣，而交易量的变化为股价未来的变化提供了线索，所以MFI指标可以帮助判断股票价格变化的趋势。
    # 策略实现
    # 1、当MFI>80，而产生背离现象时，视为卖出信号。
    # 2、当MFI<20，而产生背离现象时，视为买进信号。
    def mfi(self, datas, timeperiod=14):
        # 初始化交易信号
        tsignal = pd.Series(np.zeros((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        close = datas.Close.values
        high = datas.High.values
        low = datas.Low.values
        volume = datas.Volume.values
        # high, low, close, volume[, timeperiod=?]

        real = talib.MFI(high=high, low=low, close=close, volume=np.double(volume), timeperiod=timeperiod)

        for idx in range(30, len(datas)):
            if real[idx] > 80:
                tsignal[idx] = -1

            if real[idx ] < 20:
                tsignal[idx] = 1
        return tsignal,real

    # 阿隆指标 通过计算自价格达到近期最高值和最低值以来所经过的期间数，阿隆指标帮助你预测价格趋势到趋势区域（或者反过来，
    # 从趋势区域到趋势）的变化
    # 策略实现
    # 	1、当 AROON_UP 上穿 70，并且 AROON>0，买入，信号为 1
	#   2、当 AROON_DN 上穿 70，并且 AROON<0，卖空，信号为-1
	#   3、当 AROON_UP 下穿 50，并且 AROON<0，卖空，信号为-1
	#   4、当 AROON_DN 下穿 50，并且 AROON>0，买入，信号为 1
    def aroon(self, datas, timeperiod=14):
        tsignal = pd.Series(np.zeros((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        high = datas.High.values
        low = datas.Low.values
        aroondown, aroonup = talib.AROON(high, low, timeperiod=timeperiod)
        arron = aroonup - aroondown

        for idx in range(20, len(datas)):
            if aroonup[idx] >= 70 and aroonup[idx -1] < 70 and arron[idx] > 0:
                tsignal[idx] = 1

            if aroondown[idx] >= 70 and aroondown[idx -1] < 70 and arron[idx] < 0:
                tsignal[idx] = -1

            if aroonup[idx] < 50 and aroonup[idx -1] >= 50 and arron[idx] < 0:
                tsignal[idx] = -1


            if aroondown[idx] < 50 and aroondown[idx -1] >= 50 and arron[idx] > 0:
                tsignal[idx] = 1

        return tsignal,aroondown, aroonup

    # SAR指标又叫抛物线指标或停损转向操作点指标，其全称叫“Stop and Reverse，缩写SAR”，是由美国技术分析大师威尔斯-威尔德（Wells Wilder）所创造的，是一种简单易学、比较准确的中短期技术分析工具
    # 策略实现
    # 1、当股票股价从SAR曲线下方开始向上突破SAR曲线时，为买入信号，预示着股价一轮上升行情可能展开，投资者应迅速及时地买进股票
    # 2、当股票股价向上突破SAR曲线后继续向上运动而SAR曲线也同时向上运动时，表明股价的上涨趋势已经形成，SAR曲线对股价构成强劲的支撑，投资者应坚决持股待涨或逢低加码买进股票
    # 3、当股票股价从SAR曲线上方开始向下突破SAR曲线时，为卖出信号，预示着股价一轮下跌行情可能展开，投资者应迅速及时地卖出股票
    # 4、当股票股价向下突破SAR曲线后继续向下运动而SAR曲线也同时向下运动，表明股价的下跌趋势已经形成，SAR曲线对股价构成巨大的压力，投资者应坚决持币观望或逢高减磅
    def sar(self, datas, acceleration=0.02, maximum=0.2):
        tsignal = pd.Series(np.zeros((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        high = datas.High.values
        low = datas.Low.values
        close = datas.Close.values
        real = talib.SAR(high=high, low=low, acceleration=acceleration, maximum=maximum)
        for idx in range(10, len(datas)):
            if high[idx] > real[idx] and close[idx-1] < real[idx-1]:
                tsignal[idx] = 1

            if low[idx] < real[idx] and close[idx-1] > real[idx-1]:
                tsignal[idx] = -1

        return tsignal,real

    # 顺势指标又叫CCI指标
    # 策略实现
    # 1、当CCI曲线向上突破﹢100线进入非常态区间时，买入股票。
    # 2、当CCI曲线在﹢100线以上的非常态区间时，CCI曲线下穿CCI均线，卖出股票
    # 3、当CCI曲线在 - 100线以下的非常态区间时，CCI曲线上穿CCI均线，买入出股票。
    # 4、当CCI曲线向上突破 - 100线进入常态区时，卖出股票。
    def cci(self, datas, timeperiod=14):
        tsignal = pd.Series(np.zeros((1, len(datas)), dtype=np.int8)[0], index=datas.index)
        high = datas.High.values
        low = datas.Low.values
        close = datas.Close.values
        real = talib.CCI(high=high, low=low, close=close, timeperiod=timeperiod)

        cci_avg = talib.SMA(real, 30)
        for idx in range(50, len(datas)):
            if real[idx] > 100 and real[idx-1] < 100:
                tsignal[idx] = 1
            if real[idx -1] > 100 and real[idx] < cci_avg[idx]:
                tsignal[idx] = -1
            if real[idx -1] < -100 and real[idx] > cci_avg[idx]:
                tsignal[idx] = 1
            if real[idx] > -100 and real[idx-1] < -100:
                tsignal[idx] = -1


        return tsignal,real


class ProxyTalib(object):
    Argument_Define = {
        'MACD':['Close'],
        'MFI':['High', 'Low', 'Close', 'Volume'],
        'BBANDS':['Close'],
        'ADX':['High', 'Low', 'Close'],
        'DX':['High', 'Low', 'Close'],
        'MOM':['Close'],
        'DEMA':['Close'],
        'EMA':['Close'],
        'MA':['Close'],
        'MAMA':['Close'],
        'MIDPRICE':['High','Low'],
        'SAR':['High','Low'],
        'SMA':['Close'],
        'T3':['Close'],
        'TEMA':['Close'],
        'OBV':['Close','Volume'],
        'ADOSC':['High', 'Low', 'Close', 'Volume'],
        'AD':['High', 'Low', 'Close', 'Volume'],
        'HT_TRENDMODE':['Close'],
        'HT_SINE':['Close'],
        'HT_PHASOR':['Close'],
        'HT_DCPHASE':['Close'],
        'HT_DCPERIOD':['Close'],
        'AVGPRICE':['Open','High','Low','Close'],
        'WCLPRICE':['High','Low','Close'],
        'TRANGE':['High','Low','Close'],
        'NATR':['High','Low','Close'],
        'ATR':['High','Low','Close'],
        'WILLR':['High','Low','Close'],
        'ULTOSC':['High','Low','Close'],
        'TRIX':['Close'],
        'STOCHRSI':['Close'],
        'STOCHF':['High','Low','Close'],
        'STOCH':['High','Low','Close'],
        'RSI':['Close'],
        'ROCR100':['Close'],
        'ROCR':['Close'],
        'ROCP':['Close'],
        'ROC':['Close'],
        'PPO':['Close'],
        'PLUS_DM':['High','Close'],
        'PLUS_DI':['High','Low','Close'],
        'MOM':['Close'],
        'MINUS_DM':['High','Low'],
        'MINUS_DI':['High','Low','Close'],
        'MACDEXT':['Close'],
        'MACDFIX':['Close'],
        'CMO':['Close'],
        'CCI':['High','Low','Close'],
        'BOP':['Open','High','Low','Close'],
        'AROONOSC':['High','Close'],
        'APO':['Close'],
        'ADXR':['High','Low','Close'],
        'WMA':['Close'],
        'TRIMA':['Close'],
        'SAREXT':['High','Close'],
        'MIDPRICE':['High','Close'],
        'MIDPOINT':['Close']
    }
    #
    # Argument_Define = {
    #     'AD': ['High', 'Low', 'Close', 'Volume'],
    #     'DEMA': ['Close'],
    #     'MACDFIX': ['Close'],
    #     'BOP': ['Open', 'High', 'Low', 'Close']
    # }
    Index_Filter_Define = ['Cycle Indicators', 'Overlap Studies']

    def __init__(self):
        pass


    @staticmethod
    def proxy(data, index_keys=None):
        tret = {}
        use_inx_key = index_keys or ProxyTalib.Argument_Define.keys()
        for m in use_inx_key:
            argm = [data[x].values for x in ProxyTalib.Argument_Define[m]]
            ret = getattr(talib,m)(*argm)
            tret[m] = ret

        return tret

    # Check index data corr income
    @staticmethod
    def scorr(data):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        ixm = ProxyTalib.proxy(data)

        result = {}
        for k,v in ixm.items():
            txd = np.vstack(v)
            income = data
            if type(v) is np.ndarray:
                td = pd.DataFrame(data=txd, index=income.index, columns=list(['%s0' % k]))
            else:
                td = pd.DataFrame(data=txd.T, index=income.index,
                                  columns=list(['%s%s' % (k, x) for x in range(0, len(txd))]))
            income = pd.merge(income, td, left_index=True, right_index=True)
            #
            corr = income.corr()

            cname = income.columns.values
            cnm = [x for x in cname if x != 'INCOME' and x != 'Flag']
            formul = 'INCOME ~ {0}'.format('+'.join(cnm))
            formul = formul.replace('Volume','np.log(Volume)').replace('Amount', 'np.log(Amount)')
            rs = smf.ols(formula=formul, data=income).fit()
            print(rs.summary())

            result[k] = rs.rsquared
        return result


if __name__ == '__main__':
    from common.sdata import StockData
    data = StockData('600893').combine_income(10)
    # data.Volume = np.double(data.Volume)
    #
    # # ret = ProxyTalib.proxy(data=data)
    # ret = ProxyTalib.scorr(data)
    # print(ret)
    im = IndexModel()
    mad = im.macd(datas=data)
    mad = pd.DataFrame(mad[0], columns=['PRED'])
    data_m = pd.merge(data,mad, left_index=True, right_index=True)

    print(len(data_m[(data_m['Flag'] == 1) & (data_m['PRED'] == 1)]))










