# from html.parser import HTMLParser
import re
from bs4 import BeautifulSoup
import lxml
from lxml.html.clean import Cleaner
# from lxml.etree import HTMLParser
from lxml import etree

# https://www.cnblogs.com/zhangxinqi/p/9210211.html lxml



class WebParser:
    def __init__(self):
        super(HTMLParser, self).__init__()

    def remove_all_html_tag(self, content):
        # soup = BeautifulSoup(content,'html.parser')        
        # result = soup.get_text()
        # return result
        # cleaner = Cleaner()
        # cleaner.javascript = True # This is True because we want to activate the javascript filter
        # cleaner.style = True      # This is True because we want to activate the styles & stylesheet filter      

        cleaner = Cleaner(style=True, links=True, add_nofollow=True, page_structure=False, safe_attrs_only=False)
        content = cleaner.clean_html(content)    
        response = etree.HTML(text=content)
        result = response.xpath('string(.)')
        result = result.replace('\n\n','')
        result = result.replace('\n',',')
        content = content.replace('&nbsp;','')


        return result
 
    def get_node_content_text(self, content, xpath):
        html=etree.HTML(content,etree.HTMLParser())
        result=html.xpath(f'{xpath}/text()')
        return result



    def __call__(content):
        raise 'not implement exception '


class BaiduBaikeWebParser(WebParser):
    def __init__(self):
        super(WebParser, self).__init__()


    def __call__(self,content):
        result = self.remove_all_html_tag(content)
        return result












