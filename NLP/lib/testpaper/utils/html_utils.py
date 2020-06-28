from html.parser import HTMLParser

# https://www.cnblogs.com/liuhaidon/archive/2019/12/18/12060184.html

class PaperHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []   # 定义data数组用来存储html中的数据
        self.paragraph = ''   # 段落
        self.image_map = {}
        self.image_seq = 0
        self.recording = False 

    def handle_starttag(self, tag, attrs):
        # if self.recording and tag == 'b':
        #     self.paragraph = self.paragraph + '\n'

        if tag != 'p':
            return

        for name, value in attrs:
            if name == 'class' and value == 'cjk':
                self.recording = True
        

    def handle_endtag(self, tag):
        if tag == 'p' and self.recording:
            self.recording = False
            self.data.append(self.paragraph)
            self.paragraph = ''

    def handle_startendtag(self, tag, attrs):
        if tag != 'img':
            return
        if self.recording:
            for name, value in attrs:
                if name == 'src':
                    src = value
                    break
            self.image_seq = self.image_seq + 1
            self.image_map[self.image_seq] = src
            self.paragraph = self.paragraph + '{img:%s}' % self.image_seq

 
    def handle_data(self, data):
        data = data.replace('\n','')
        if self.recording:
            self.paragraph = self.paragraph + ' ' + data.strip()


def parse_paper(file_name):
    print('file name:', file_name)
    with open(file_name, 'r') as f:
        content = f.read()

    parser = PaperHTMLParser()
    parser.feed(content)
    parser.close()
    print('\n'.join(parser.data))
    return parser.data, parser.image_map

    



if __name__ == '__main__':
    parse_paper(u'D:\\PROJECT_TW\\git\\data\\testpaper\\output\\207雅礼高一上第三次月考-教师版.html')