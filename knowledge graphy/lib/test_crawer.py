from kg_tools.crawer.web_crawer import BaiduBaikeCrawer
from kg_tools.crawer.html_parser import BaiduBaikeWebParser
from os.path import join

def test_baidubaike_crawer(data_dir, content):
    crawer = BaiduBaikeCrawer()
    resp = crawer(content)
    with open(join(data_dir, 'baidubk', f'{content}.txt'), 'w', encoding='utf-8')  as f:
        f.write(resp.text)
    print('crawer over ')

def test_baidubaike_parser(data_dir, file_name):
    parser = BaiduBaikeWebParser()
    content = None
    with open(join(data_dir, 'baidubk', f'{file_name}.txt'), 'r', encoding='utf-8')  as f:
        content = f.read()
    if content:
        # 得到所有
        content = parser(content)
        
        #  带条件查找
        # content = parser.get_node_content_text(content, xpath='//div')
        # content = [x.strip() for x in content if x != '\n']
        # content = ','.join(content)

    print('content', content)
    return content


if __name__ == '__main__':
    import argparse
    import logging
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", default=r'D:\PROJECT_TW\git\data\kg\crawer', help="", type=str)
    args = parser.parse_args()

    content = '平行四边形'
    # test_baidubaike_crawer(data_dir=args.data_dir,content=content)

    test_baidubaike_parser(data_dir=args.data_dir, file_name=content)