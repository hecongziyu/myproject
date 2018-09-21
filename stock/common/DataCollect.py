# -*- coding: utf-8 -*-
# 采集数据 "http://money.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/:stockcode.phtml?year=:year&jidu=";
# Python 与 R 通过 R RServer进行交互
import requests
import re

# 定义script的正则表达式
regEx_script = "<script[^>]*?>[\\s\\S]*?<\\/script>"
# 定义style的正则表达式
regEx_style = "<style[^>]*?>[\\s\\S]*?<\\/style>"
# 定义HTML标签的正则表达式
regEx_html = "<[^>]+>"
# 定义空格回车换行符
regEx_space = "\t"
# 定义空格回车换行符
regEx_rtspace = "\r|\n"


class DataCollect(object):
    def __init__(self, data_path, **kw):
        self.name = 'data collect'
        self.data_url = ('http://money.finance.sina.com.cn/corp/go.php/'
                         'vMS_MarketHistory/stockid/:stockcode.phtml?year=:year&jidu=')
        self.data_base_path = data_path

    def collect_his(self, code):
        alldata = []

        for year in range(2005, 2019):
            url = self.data_url.replace(':stockcode', code).replace(':year', str(year))
            print('down load url :' + url)
            for index in range(1, 5):
                rurl = url + str(index)
                content = HttpUtils.getdatafromweb(rurl)
                detail = self.parse_content(content, str(year))
                alldata = alldata + detail
        self.savetofile(code, code + '.txt', alldata)


    def savetofile(self, dirname, filename, content):
        dirpath = self.data_base_path + os.path.sep + dirname
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        filepath = dirpath + os.path.sep + filename
        if isinstance(content, list):
            content = sorted(content)
            # 插入标头
            content.insert(0, 'Date,Open,High,Close,Low,Volume,Amount,Extend')
            file = open(filepath, 'w+')
            for line in content:
                line = line.strip('\n')     # 避免原字符串里面有换行标记
                file.write(line + '\n')
            file.close()
        else :
            raise Exception('需保存文件类型不是LIST类型，请检查')

    def readfromfile(self, code):
        filepath = self.data_base_path + os.path.sep + code + os.path.sep + code + '.txt'
        print(filepath)
        file = open(filepath, 'r')
        lines = file.readlines()
        file.close()
        return lines

    @staticmethod
    def parse_content(content, tag):
        begin = content.find('<table id=\"FundHoldSharesTable\">')
        content = content[begin + len('<table id=\"FundHoldSharesTable\">'):]
        end = content.find('</table>')
        content = content[0:end]
        content = content.replace('<tr>', '')
        content = content.replace('</tr>', '|')
        content = re.sub(regEx_script, '', content)
        content = re.sub(regEx_style, '', content)
        content = re.sub(regEx_html, '', content)
        content = re.sub(regEx_space, '', content)
        content = re.sub(regEx_rtspace, ',', content)
        content = content.replace(' ', '')
        content = content.replace(',,,,,,,,', '')
        content = content.replace(',,,,', ',')
        content = content.replace(',,', ',')
        content = content.replace(',,2', '2')
        content = content.split('|')
        detail = []
        for line in content:
            if line.find(tag) != -1:
                detail.append(line)
        return detail


class HttpUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def getdatafromweb(url, method='GET'):
        respone = requests.get(url)
        return respone.content


if __name__ == '__main__':
    import os
    print(__file__)
    print("os.path.dirname(os.path.realpath(__file__))=%s" % os.path.dirname(os.path.realpath(__file__)))
    dc = DataCollect(data_path='D:\\PROJECT_TW\\analysis\\data')
    dc.collect_his("601818")
    #dc.savetofile('600016', '600016.txt', ['1111', '222'])
    #cl = dc.readfromfile('600016')


