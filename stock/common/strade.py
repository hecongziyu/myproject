# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# 交易模拟

class StockTrade(object):
    def __init__(self):
        pass

    # trade_data 交易数据，  trade_signal 交易信号
    # cash 交易初始化金额
    # p 487
    def trade(self, init_cash, trade_data, trade_signal):
        asset = pd.Series(0.0, index=trade_data.index)
        cash = pd.Series(0.0, index=trade_data.index)
        share = pd.Series(0,index=trade_data.index)

        # 当价格连续两天上涨且交易信号没有卖出标志时，第一次持有股
        entry = 3
        cash[:entry] = init_cash
        while entry < len(trade_data):
            cash[entry] = cash[entry-1]
            if all([trade_data.Close[entry-1] >= trade_data.Close[entry-2],
                    trade_data.Close[entry-2] >= trade_data.Close[entry-3],
                    trade_signal[entry-1] != -1]):
                share[entry] = cash[entry]/trade_data.Close[entry]
                cash[entry] = cash[entry] - share[entry] * trade_data.Close[entry]
                break
            entry += 1

        i = entry + 1
        while i < len(trade_signal):
            cash[i] = cash[i-1]
            share[i] = share[i-1]

            if all([trade_signal[i] == 1,cash[i]>trade_data.Close[i]]):
                share[i] = share[i] + cash[i]/trade_data.Close[i]
                cash[i] = cash[i] - share[i] * trade_data.Close[i]


            if all([trade_signal[i] == -1, share[i] > 0]):
                cash[i] = cash[i] + trade_data.Close[i] * share[i]
                share[i] = 0


            i += 1

        asset = cash + share*trade_data.Close
        # 官方推荐，assign 为DataFrame增加新列。
        # trade_data.loc[:, 'Asset'] = asset.tolist()
        # trade_data.loc[:, 'Share'] = share.tolist()
        # trade_data.loc[:, 'Cash'] = cash.tolist()
        trade_data = trade_data.assign(Asset=asset.values, Share=share.values, Cash=cash.values)
        return trade_data


if __name__ == '__main__':
    from common.sdata import StockData
    sd = StockData(code='600893')
    sdatas = sd.combine_income(ndays=5)
    sdatas = sdatas.tail(100)
    sradom = np.random.choice(a=[-1, 0, 1], size=len(sdatas), replace=True, p=[0.3, 0.4, 0.3])
    signal = pd.Series(sradom, index=sdatas.index)

    st = StockTrade()
    tradeData = st.trade(20000, trade_data=sdatas, trade_signal=signal)
    tradeData = tradeData[['Close','Share','Cash','Asset']]
    tradeData = tradeData.assign(signal=signal)
