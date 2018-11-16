# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
import re
import os

#https://jingyan.baidu.com/article/3065b3b68d7fb5becff8a494.html uShare是一个免费、开源的python财经数据接口包。主要实现对股票等金融数据从数据采集、清洗加工到数据存储的过程，能够为金融分析人员提供快速、整洁、和多样的便于分析的数据

# 数据加载部分

# 0 sell 1 hold 2 buy

class StockData(object):
    def __init__(self,code):
        self.code = code
        self.datas = self.__read_data__(code)

    def __read_data__(self, code):
        df = pd.read_table('D:/PROJECT_TW/anly/data/'+code+'/' + code + '.txt', sep=',', header=0, index_col='Date')
        df.index = pd.to_datetime(df.index)
        del df['Extend']
        return df

    # get n days income flag
    def combine_income(self, ndays=1, sell=-1, buy=1):
        
        Close = self.datas.Close
        Low = self.datas.Low
        High = self.datas.High
        
        columns = []
        for i in range(ndays):
            shift_num = 0 - i - 1
            columns.append('IMCOME_'+str(i+1))
            self.datas['IMCOME_'+str(i+1)] =  ((((Close.shift(shift_num).values + Low.shift(shift_num).values + High.shift(shift_num).values)/3)/Close) -1) * 100
        self.datas = self.datas.dropna(axis=0,how='any')        # how = any 指只要出现一个就删除  = all 是只有所有为空进删除
        self.datas['INCOME'] = self.datas[columns].max(axis=1)
#         for key in columns:
#             del self.datas[key]

        self.datas['Flag'] = list(map(lambda x: 0 if x <= sell else(1 if x > sell and x <= buy  else 2),self.datas.INCOME))
        return self.datas


    # 将数据打包成时间序列形式数据，每行数据包括前pdyas天的数据，用于lstm train
    @staticmethod
    def package_data(data, pdays=5):
        dx = []
        dy = []
        # data.Volume = data.Volume.apply(np.log)
        # data.Amount = data.Amount.apply(np.log)
        cm = data.columns.get_level_values(0)
        for i in range(pdays, len(data)+1):
            # dx.append(np.apply_along_axis(pre.scale, 1, np.array(data.iloc[(i-pdays):i, (cm!='INCOME')&(cm!='Flag')])))
            dy.append(data.iloc[i-1, data.columns == 'Flag'])
            # dy.append(data.iloc[i - 1, data.columns == 'INCOME'])
            # dx.append(data.iloc[(i - pdays):i, (cm != 'INCOME') & (cm != 'Flag')])
            dx.append(data.iloc[(i - pdays):i, (cm != 'INCOME') & (cm != 'Flag')])
            # dx.append(data.iloc[(i - pdays):i, cm == 'Close'])
            # dy.append(data.iloc[i-1, data.columns == 'Flag'])
        return dx, dy



    # 增加指标数据
    @staticmethod
    def combine_index_data(data, index_keys=None):
        from .indexModel import ProxyTalib as pt
        data.Volume = np.double(data.Volume)
        data.Amount = np.double(data.Amount)
        idx_data = pt.proxy(data, index_keys=index_keys)
        ad = data.copy()
        for ik in idx_data.keys():
            txd = np.vstack(idx_data[ik])
            if type(idx_data[ik]) is np.ndarray:
                td = pd.DataFrame(data=txd, index=data.index, columns=list(['%s0' % ik]))
            else:
                td = pd.DataFrame(data=txd.T, index=data.index,
                                  columns=list(['%s%s' % (ik, x) for x in range(0, len(txd))]))
            ad = pd.merge(ad, td, left_index=True, right_index=True)
            ad = ad.dropna()

        return ad

    @staticmethod
    def load_third_data(fileName, destFileName):
        from struct import unpack
        ofile = open(fileName, 'rb')
        wfile = open(destFileName,'w')
        buf = ofile.read()
        ofile.close()
        num = len(buf)
        no = num / 32
        b = 0
        e = 32
        items = list()
        wfile.write('Date,Open,High,Close,Low,Volume,Amount,Extend\n')
        for i in range(int(no)):
            a = unpack('IIIIIfII', buf[b:e])
            year = int(a[0] / 10000);
            m = int((a[0] % 10000) / 100);
            month = str(m);
            if m < 10:
                month = "0" + month;
            d = (a[0] % 10000) % 100;
            day = str(d);
            if d < 10:
                day = "0" + str(d);
            dd = str(year) + "-" + month + "-" + day
            openPrice = a[1] / 100.0
            high = a[2] / 100.0
            low = a[3] / 100.0
            close = a[4] / 100.0
            amount = a[5] / 10.0
            vol = a[6]
            unused = a[7]
            if i == 0:
                preClose = close
            ratio = round((close - preClose) / preClose * 100, 2)
            preClose = close
            item = [dd, str(openPrice), str(high), str(close), str(low), str(vol), str(amount)]
            items.append(item)
            item_s = ','.join(item)
            wfile.write(item_s+',\n')
            b = b + 32
            e = e + 32
        wfile.close()
        return items

    @staticmethod
    def load_all_data(src_dir, dest_dir):
        files = [x for x in os.listdir(src_dir) if x.endswith('.day')]
        for file in files:
            src_file = os.path.sep.join([src_dir, file])
            dest_file_name =  str(re.findall('\d+',file)[0])
            dest_file = os.path.sep.join([dest_dir,dest_file_name])
            if not os.path.exists(dest_file):
                os.mkdir(dest_file)
            dest_file = os.path.sep.join([dest_file,dest_file_name + '.txt'])
            StockData.load_third_data(src_file, dest_file)
            print('src {} dest {}'.format(src_file, dest_file))


if __name__ == '__main__':
    # sd = StockData(code='600016')
    # datas = sd.combine_income(ndays=5)
    # datas = datas.head(50)
    #
    # pdata = StockData.package_data(datas)
    # x ,y = pdata[0], pdata[1]
    # import torch
    # x = [z.values for z in x]
    # xt, yt = torch.from_numpy(np.array(x)).float(), torch.from_numpy(np.array(y)).float()
    # items = StockData.load_third_data('../data/sh600016.day','../data/sh600016.txt')
    StockData.load_all_data('C:\\new_tdx\\vipdoc\\sz\\lday','D:\\PROJECT_TW\\anly\\data')
    # pass

