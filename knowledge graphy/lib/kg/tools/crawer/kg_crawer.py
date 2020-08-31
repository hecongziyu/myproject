# 爬虫
# https://blog.csdn.net/qq_41661056/article/details/95788729  生成浏览器的User-Agent信息之fake_useragent
# https://blog.csdn.net/qq_40147863/article/details/81710220?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
# https://blog.csdn.net/weixin_43215588/article/details/102567210  微博爬虫
# https://blog.csdn.net/u011909077/article/details/100145910 !
from selenium import webdriver
import time

class Crawer:
    def __init__(self):
        # 加入opthon 设置可登录一次后长期可用
        self.option = webdriver.ChromeOptions()
        self.option.add_argument('--user-data-dir=C:\\Users\\Administrator\\AppData\\Local\\Google\\Chrome\\User Data L')
        # option.add_argument('--headless') # 增加无界面选项
        # option.add_argument('--disable-gpu') # 如果不加这个选项，有时定位会出现问题
        

    def __call__(self):
        raise 'not implement exception '


# https://s.weibo.com/weibo?q=%E6%8B%93%E7%BB%B4%E4%BF%A1%E6%81%AF&wvr=6&Refer=SWeibo_box
class CrawerWeiboSearch(Crawer):
    '''
    通过微博查询，爬取信息
    '''
    def __init__(self):
        Crawer.__init__(self)
        self.url = 'https://weibo.com/'

    def __call__(self, content):
        texts = []
        try:
            self.web=webdriver.Chrome(options=self.option)
            self.web.get('https://weibo.com/')
            self.web.implicitly_wait(30)
            # 输入查询条件
            self.web.find_element_by_class_name('W_input').send_keys('拓维信息')
            # 搜索按扭
            self.web.find_element_by_class_name('ficon_search').click()
            self.web.implicitly_wait(30)

            # 后期修改，检测是否有该元素
            # self.web.find_element_by_id('loginname').send_keys('13873153921@139.com')
            # time.sleep(1)
            # self.web.find_element_by_name('password').send_keys('home44292508')
            # time.sleep(1)
            # self.web.find_element_by_xpath("//div[@class='W_login_form']//div[6]//a[1]").click()            
            # self.web.implicitly_wait(30)

            # 注意查找多个元素用find_elements, 单个元素用find_element
            nodes = self.web.find_elements_by_class_name('content')

            # 暂时不需要，因为登录信息已保存
            # web.find_element_by_id('loginname').send_keys('账号')
            # time.sleep(1)
            # web.find_element_by_name('password').send_keys('密码')
            # time.sleep(1)
             
            # web.find_element_by_xpath("//div[@class='W_login_form']//div[6]//a[1]").click()            

            for item in nodes:
                texts.append(item.text)
        finally:
            self.web.quit()
        return texts



if __name__ == '__main__':
    from os.path import join
    crawer = CrawerWeiboSearch()
    texts = crawer('拓维信息')

    with open(join(r'D:\PROJECT_TW\git\data\finance', 'craw.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))










