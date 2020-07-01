# -*- coding: UTF-8 -*-
from config import _C as cfg, logger
import argparse
from utils.docx_utils import convert_docx
from utils.html_utils import parse_html_paper
from txtdataset import TARGETS,STOP_WORDS,TAG_QUESTION,TAG_CONTENT,TAG_ANSWER,TAG_ANSWER_AREA
import os 
import shutil
from paper_detect import PaperDetect


class PaperQuestion(object):
    def __init__(self, qid, parent_qid):
        self.qid = qid
        self.parent_qid = parent_qid
        # 对应问题位置区域, 位置区域包含子问题
        self.postion = (-1,-1)
        # 问题内容
        self.content = None

    def __repr__(self):
        return '{%s:%s %s %s }' % (self.parent_qid, self.qid, self.postion, self.content)

    def __str__(self):
        return '{%s:%s %s %s }' % (self.parent_qid, self.qid, self.postion, self.content)


def clean_env(data_path):
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)    

def paper_parse_content(file_name):
    clean_env(cfg.paper.temp_path)
    logger.info('开始处理 %s,  转换成HTML' % file_name)
    convert_docx(cfg.server.libreoffice, os.path.sep.join([cfg.paper.data_root,file_name]), cfg.paper.temp_path)
    logger.info('转换成HTML，开始解析试卷HTML内容')
    pcontent, pimage = parse_html_paper(os.path.sep.join([cfg.paper.temp_path, '%s.html' % file_name.rsplit('.',1)[0]]))
    logger.debug('解析后的HTML内容：%s' % '\n'.join(pcontent))
    return pcontent


def paper_split_content(pdetect, lines, parent_id=None):
    '''
    处理流程：
    1、检测 line 类型 （问题、答案、解析、内容、答案区域开始, 备注）
    2、提取分词, 对第一个分词进行问题级别分类，根据当前问题级别对该行进行问题分级
    3、到第一步

    保存结构：
    Question:
        id:  ID号
        parent_id: 上级ID号
        content: 问题内容  (后期再进行分解成答案、问题等部分)
    '''
    # print('paper split content ....', lines)
    lines = [x.strip() for x in lines]
    qlists = []
    q_start_flag = False
    q_start_idx = None
    q_end_idx = None
    current_level = None
    question = None
    for idx, line in enumerate(lines):
        ltype, qlevel = pdetect.detect(line)
        if ltype == TAG_QUESTION:
            if not q_start_flag:
                current_level = qlevel
                q_start_idx = idx
                q_start_flag = True
                question = PaperQuestion(qid=q_start_idx, parent_qid=parent_id)
                question.content = line
            else:
                if current_level == qlevel:
                    q_end_idx = idx
                    question.postion = (q_start_idx, q_end_idx)
                    qlists.append(question)
                    if q_end_idx - q_start_idx > 1:
                        sub_qlists = paper_split_content(pdetect, lines[q_start_idx+1:q_end_idx], question.qid)
                        if len(sub_qlists) > 0:
                            qlists.extend(sub_qlists)
                    q_start_idx = idx
                    question = PaperQuestion(qid=q_start_idx, parent_qid=parent_id)
                    question.content = line
        else:
            if question is not None:
                question.content = '%s %s' % (question.content, line)


    if question is not None:
        q_end_idx = len(lines)
        question.postion = (q_start_idx, q_end_idx)
        qlists.append(question)
        if q_end_idx - q_start_idx > 1:
            sub_qlists = paper_split_content(pdetect, lines[q_start_idx+1:q_end_idx], question.qid)
            if len(sub_qlists) > 0:
                qlists.extend(sub_qlists)

    return qlists




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="试卷导入功能")
    parser.add_argument("--config_file", default="bootstrap.yml", help="配置文件路径", type=str)
    parser.add_argument("--file_name", default=u"207雅礼高一上第三次月考-教师版.docx", help="配置文件路径", type=str)
    parser.add_argument("--data_root", default="D:\\PROJECT_TW\\git\\data\\testpaper", help="配置文件路径", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)    
    # paper_parse_content(args.file_name)

    with open(os.path.sep.join([args.data_root,'simulate', 'paper_1.txt']), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pdetect = PaperDetect()
    qlists = paper_split_content(pdetect, lines)

    print('\n'.join([str(x) for x in qlists]))






