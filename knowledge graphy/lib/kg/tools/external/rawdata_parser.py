# 原始数据解析
from html.parser import HTMLParser


class RawDataParser:
    def __init__(self):
        pass


    def __call__(self, raw_data):
        raise Exception('call method not implement')



class HtmlDataParser(RawDataParser,HTMLParser):
    def __init__(self):
        RawDataParser.__init__(self)
        HTMLParser.__init__(self)
        self.data = []
        self.begin_flag = False
        self.start_flag = False


# 新浪半规则化股票行业HTML解析
class HtmlSinaBelongToParser(HtmlDataParser):
    def __init__(self):
        HtmlDataParser.__init__(self)

    def __call__(self, raw_data):
        self.feed(raw_data)
        self.close()
        return self.data

    def handle_starttag(self, tag, attrs):
        if tag != 'td':
            return

        for name, value in attrs:
            if name == 'class' and value == 'ct':
                self.begin_flag = True
                break
    
    def handle_endtag(self, tag):
        if tag == 'table':
            self.begin_flag = False
            self.start_flag = False

    def handle_data(self, raw_data):
        raw_data = raw_data.strip()
        if self.begin_flag:
            if self.start_flag and raw_data not in ['点击查看','概念板块','同概念个股'] and len(raw_data)>0:
                self.data.append(raw_data)
            elif raw_data == '所属概念板块':
                self.start_flag = True







