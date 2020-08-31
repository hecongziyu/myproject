# import sys
# sys.path.append('D:\\PROJECT_TW\\git\\finance')
import lib.utils.http_utils as http_utils
from lib.kg.rawdata_parser import *
# 第三方接口

class KGDataGather:
    '''
    知识图谱数据采集
    '''
    def __init__(self, rel_name=None):
        self.rel_name = rel_name



class KGRuleDataSinaGather(KGDataGather):
    def __init__(self, rel_name):
        super(KGRuleDataSinaGather, self).__init__(rel_name)
        self.rel_url_map = {
            'BelongTo':(HtmlSinaBelongToParser(), 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpOtherInfo/stockid/{}/menu_num/2.phtml')
        }



    def __call__(self, code, method='GET'):
        # print('rel name:', self.rel_name)
        parser, url = self.rel_url_map[self.rel_name]
        url = url.format(code)
        if method == 'GET':
            rsp_content = http_utils.do_get(url)
            rel_entity = parser(rsp_content)
        return self.rel_name, rel_entity



if __name__ == '__main__':
    # rel = KGRuleDataSinaGather('BelongTo')
    # rel_info = rel(code='002261')

    get_entity_by_name('拓维信息')



