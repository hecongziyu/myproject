from html.parser import HTMLParser
import chardet

# https://www.cnblogs.com/liuhaidon/archive/2019/12/18/12060184.html
# https://www.cnblogs.com/schut/p/10579955.html 检测文件编码

class PaperHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []   # 定义data数组用来存储html中的数据
        self.paragraph = ''   # 段落
        self.image_map = {}
        self.image_seq = 0
        self.recording = 0
        self.start_num_map = {}
        self.ol_start_value = None


    def handle_starttag(self, tag, attrs):
        # if self.recording and tag == 'b':
        #     self.paragraph = self.paragraph + '\n'

        if tag not in ['p','table', 'tr','td','ol','li','h1']:
            return

        # for name, value in attrs:
        #     if name == 'class' and value == 'cjk':
        if tag in ['p','table','h1']:
            self.recording = self.recording + 1

        # if self.recording > 0:
        if tag == 'ol':
            # ol libreoffice 解析时  <ol start=17></ol>  next <ol start=17></ol> ,数字相同
            self.ol_start_value = None
            for name, value in attrs:
                if name == 'start':
                    self.ol_start_value = value
                    if value in self.start_num_map:
                        self.start_num_map[value] = self.start_num_map[value] + 1
                    else:
                        self.start_num_map[value] = int(value)
                    break

        # elif tag == 'li':
        #     # if self.ol_start_value  is not None:
        #         # self.paragraph = self.paragraph + f'{self.start_num_map[self.ol_start_value]}.'
        #         # pass
        #     # else:
        #     self.paragraph = self.paragraph + '[OL]'
            
        
        if tag in ['table','tr','td']:
            self.paragraph = self.paragraph + f'<{tag}>'
        

    def handle_endtag(self, tag):
        if self.recording > 0:
            if tag in ['p','table','h1']:
                self.recording = self.recording - 1

            if tag in ['p','h1']:
                self.data.append(self.paragraph)
                self.paragraph = ''
            elif tag in ['table','tr','td']:
                self.paragraph = self.paragraph + f'</{tag}>'
                if tag == 'table':
                    self.paragraph = self.paragraph + '\n'
            elif tag == 'OL':
                self.ol_start_value = None

    def handle_startendtag(self, tag, attrs):
        if tag != 'img':
            return
        if self.recording > 0:
            src = None
            for name, value in attrs:
                if name == 'src':
                    src = value
                    break
            if src is not None:
                self.image_seq = self.image_seq + 1
                self.image_map[self.image_seq] = src
                self.paragraph = self.paragraph + '{img:%s}' % self.image_seq

 
    def handle_data(self, data):
        data = data.replace('\n','')
        if self.recording > 0:
            self.paragraph = self.paragraph + data.strip()


def parse_html_paper(file_name):
    print('file name:', file_name)
    result = check_file_character(file_name)
    print('character:', result)
    encode = 'utf-8' if result['encoding'] == 'utf-8' else 'GBK'
    with open(file_name, 'r', encoding=encode, errors='ignore') as f:
        content = f.read()

    parser = PaperHTMLParser()
    parser.feed(content)
    parser.close()
    # print('\n'.join(parser.data))
    return parser.data, parser.image_map

def check_file_character(file_name):
    with open(file_name, 'rb') as f:
        data = f.read()
        result = chardet.detect(data)
    return result



    



if __name__ == '__main__':
    parse_html_paper(u'D:\\PROJECT_TW\\git\\data\\testpaper\\html\\207雅礼高一上第三次月考-教师版.html')