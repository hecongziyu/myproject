# -*- coding: UTF-8 -*-
import os
import argparse
from config import _C as cfg, logger
from utils.docx_utils import convert_docx_to_html, convert_docx_to_text
from utils.html_utils import parse_html_paper
import Levenshtein
import re
import numpy as np
from utils.txt_utils import txt_ratio, combine_include_img_str

def paper_doc_convert(file_name):
    logger.info('开始处理 %s,  转换成TEXT' % os.path.sep.join([cfg.paper.data_root,file_name]))
    convert_docx_to_text(cfg.server.libreoffice, os.path.sep.join([cfg.paper.data_root,file_name]), cfg.paper.temp_path)
    logger.info('开始处理 %s,  转换成HTML' % os.path.sep.join([cfg.paper.data_root,file_name]))
    convert_docx_to_html(cfg.server.libreoffice, os.path.sep.join([cfg.paper.data_root,file_name]), cfg.paper.temp_path)
    logger.info('转换成HTML，开始解析试卷HTML内容')
    pcontent, pimage = parse_html_paper(os.path.sep.join([cfg.paper.temp_path, '%s.html' % file_name.rsplit('.',1)[0]]))
    logger.info('解析后的HTML内容：%s' % '\n'.join(pcontent))
    acontent = adjuest_paper_content(file_name, pcontent)
    logger.info('TXT与HTML融合后的内容：%s' % '\n'.join(acontent))

    return acontent, pimage





# 整理HTML内容，与生成的txt文档进行比较，进行整合, 将相似度大的
def adjuest_paper_content(file_name, hcontents):
    with open(os.path.sep.join([cfg.paper.temp_path,'%s.txt' % file_name.rsplit('.',1)[0]]),'r', encoding='utf-8') as f:
        txtcnts = f.readlines()
    a_contents = []
    current_t_idx = 0
    
    for hitem in hcontents:
        flag_sim = False
        for cidx in range(current_t_idx, len(txtcnts)):
            jratio = txt_ratio(hitem, txtcnts[cidx])
            if jratio >= 0.8:
                current_t_idx = cidx
                flag_sim = True
                break

        # print('cur idx:', current_t_idx,'(', flag_sim ,'):', jratio, ': content ->', txtcnts[cidx], ': hcontents ->' , hitem)


        if flag_sim:
            _contents = combine_include_img_str(hitem,txtcnts[current_t_idx].replace('\n','').strip())
            current_t_idx = current_t_idx + 1
            # _contents = txtcnts[current_t_idx].replace('\n','').strip()
            a_contents.append(_contents)
        else:
            a_contents.append(hitem)


    return a_contents


def gen_train_data(data_dir):
    file_lists = os.listdir(data_dir)
    for idx, item in enumerate(file_lists):
        content = paper_doc_convert(item)
        with open(os.path.sep.join([cfg.paper.ouput_path, '%s.txt' % item.rsplit('.',1)[0]]),'w', encoding='utf-8') as f:
            f.writelines([ '%s\n' % x for x in  content])
        logger.info('处理[ %s ] 完成' % item)  

def gen_train_data_file(file_name):       
    content, image = paper_doc_convert(file_name)
    with open(os.path.sep.join([cfg.paper.ouput_path, '%s.txt' % file_name.rsplit('.',1)[0]]),'w', encoding='utf-8') as f:
        f.writelines([ '%s\n' % x for x in  content])
    with open(os.path.sep.join([cfg.paper.ouput_path, '%s_img.txt' % file_name.rsplit('.',1)[0]]),'w', encoding='utf-8') as f:
        f.writelines(['%s\n' %x for x in image])        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="试卷导入功能")
    parser.add_argument("--config_file", default="bootstrap.yml", help="配置文件路径", type=str)
    parser.add_argument("--file_name", default=u"2016年秋季长沙市一中高一期中考试试卷--教师版.docx", help="配置文件路径", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)    
    print(cfg.paper.ouput_path)
    gen_train_data_file(args.file_name)

    # str1 = '{img:0}{img:3}当{img:5}时，{img:6}证明；{img:1}{img:2}'
    # str1 = '当{img:5}时，{img:6}{img:1}{img:2}'
    # str2 = '当时，证明；'
    # str1 = '当时，{img:0}A{img:1} {img:2}'
    # str2 = '(1) 当时，证明；'
    # str2 = '(1) 当时'

    # str1 = '{img:1}{img:5}{img:6}{img:7}'
    # str2 = ''
    # str1 = '{img:24}{img:25}综上{img:26}'
    # str2 = '综上'

    # str1 = '{img:21}，{img:22}又{img:23}'
    # str2 = '，又'

    # str1 = '（2）若{img:22}…{img:23}=1，则{img:24}…{img:28}{img:29}…{img:32}。'
    # str2 = '（2）若…=1，则……。'

    # str1 = '{img:11}或{img:0},{img:1},{img:2}'
    # str2 = '或,,'
    # result = combine_include_img_str(str1,str2)
    # print('%s -- %s --> %s' % (str1, str2, result))
