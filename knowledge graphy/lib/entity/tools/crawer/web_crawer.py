# 爬虫
# https://blog.csdn.net/qq_41661056/article/details/95788729  生成浏览器的User-Agent信息之fake_useragent
# https://blog.csdn.net/qq_40147863/article/details/81710220?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
# https://blog.csdn.net/weixin_43215588/article/details/102567210  微博爬虫
# https://blog.csdn.net/u011909077/article/details/100145910 !
from selenium import webdriver
import time
import urllib.parse
import requests

class Crawer:
    def __init__(self):
        # 加入opthon 设置可登录一次后长期可用
        # self.option = webdriver.ChromeOptions()
        # self.option.add_argument('--user-data-dir=C:\\Users\\Administrator\\AppData\\Local\\Google\\Chrome\\User Data L')
        # option.add_argument('--headless') # 增加无界面选项
        # option.add_argument('--disable-gpu') # 如果不加这个选项，有时定位会出现问题
        pass        
    def __call__(self,url):
        raise 'not implement exception '



class BaseCrawer(Crawer):
    def __init__(self):
        super(BaseCrawer,self).__init__()
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}        

    def __invoke__(self,url):
        resp = requests.get(url, headers=self.headers)  
        return resp      



class BaiduBaikeCrawer(BaseCrawer):
    def __init__(self):
        super(BaiduBaikeCrawer,self).__init__()
        self.pre_url = 'https://baike.baidu.com/item/'

    def __call__(self, content):
        url = self.pre_url + urllib.parse.quote(content)
        return self.__invoke__(url)



class ZhiShiCrawer(BaseCrawer):
    def __init__(self):
        super(ZhiShiCrawer, self).__invoke__()
        self.pre_url = 'http://zhishi.me/api/entity/'

    def __call__(self, content):
        url = self.pre_url + urllib.parse.quote(content)
        return self.__invoke__(url)
