# -*- coding: UTF-8 -*-
from main import paper_parse_content
import os
import argparse
from config import _C as cfg, logger

def gen_train_data():
    file_lists = os.listdir(cfg.paper.data_root)
    for idx, item in enumerate(file_lists):
        content = paper_parse_content(item)
        with open(os.path.sep.join([cfg.paper.ouput_path, '%s.txt' % item.rsplit('.',1)[0]]),'w', encoding='utf-8') as f:
            f.writelines([ '%s\n' % x for x in  content])
        logger.info('处理[ %s ] 完成' % item)         


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="试卷导入功能")
    parser.add_argument("--config_file", default="bootstrap.yml", help="配置文件路径", type=str)
    parser.add_argument("--file_name", default=u"207雅礼高一上第三次月考-教师版.docx", help="配置文件路径", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)    
    gen_train_data()