# 通过test paper 生成训练数据, 解析试卷word文档，分离出图片与公式图片，手工标注图片，并删除一些图片
import sys
# sys.path.append('D:\\PROJECT_TW\\git\\myproject\\NLP\\lib\\testpaper')
sys.path.append('D:\\PROJECT_TW\\git\\myproject\\NLP\\lib')
from testpaper.utils.docx_utils import convert_docx_to_html,convert_docx_to_text
from testpaper.utils.html_utils import parse_html_paper
from testpaper.utils.txt_utils import txt_ratio, combine_include_img_str
from testpaper.config import _C as cfg, logger
import os
import shutil
import argparse
import json
from pylatex.base_classes import Environment, CommandBase, Arguments
from pylatex.package import Package
from pylatex import Document, Section, UnsafeCommand
from pylatex.utils import NoEscape
import re
from PIL import Image
from pdfutils import gen_latex_img_pos
import cv2
import numpy as np
import lmdb
import random
from os.path import join
import six
from matplotlib import pyplot as plt


GEN_FILE_IDX = 0

# 预准备数据， 将原WORD文档转成HTML，并产生相应的图片信息, 同时需手工清理不需要的图片
def prepare_train_data(source_dir, dest_dir):
    file_lists = os.listdir(source_dir)
    # 修改文档名，并复制到dest dir 目录下
    for idx, fitem in enumerate(file_lists):
        
        if not os.path.exists(os.path.sep.join([dest_dir, 'use', str(idx)])):
            os.mkdir(os.path.sep.join([dest_dir,'use' ,str(idx)]))
        file_ext = fitem.rsplit('.',1)[1]
        dest_file_name = os.path.sep.join([dest_dir,'all', '%s.%s' % (str(idx), file_ext)])
        shutil.copyfile(os.path.sep.join([source_dir, fitem]), dest_file_name)
        print('开始处理：', fitem)
        convert_docx_to_text(cfg.server.libreoffice, dest_file_name, os.path.sep.join([dest_dir,'use' ,str(idx)]))
        convert_docx_to_html(cfg.server.libreoffice,dest_file_name, os.path.sep.join([dest_dir,'use' ,str(idx)]), embed_img=False)


# 生成训练数据，包含图片及位置信息
def prepare_train_images(data_dir):
    file_lists = os.listdir(os.path.sep.join([data_dir,'use']))
    print(file_lists)
    for idx, fitem in enumerate(file_lists):
        print('正在处理：', fitem, ' 生成图片数据！')
        pcontent, pimage = parse_html_paper(os.path.sep.join([data_dir,'use' ,fitem, '{}.html'.format(fitem)]))
        with open(os.path.sep.join([data_dir,'use' ,fitem, '{}_img.json'.format(fitem)]), 'w', encoding='utf-8') as f:
            f.write(json.dumps(pimage))

        acontent = adjuest_paper_content(os.path.sep.join([data_dir, 'use', fitem, '{}.txt'.format(fitem)]), pcontent)

        with open(os.path.sep.join([data_dir, 'use', fitem, '{}_out.txt'.format(fitem)]), 'w', encoding='utf-8') as f:
            acontent = '\n'.join(acontent)
            f.write(acontent)


# 整理HTML内容，与生成的txt文档进行比较，进行整合, 将相似度大的
def adjuest_paper_content(file_name, hcontents):
    with open(file_name,'r', encoding='utf-8') as f:
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


# 读取试卷question内容，生成PDF文件，并得到试卷中公式、图片的位置坐标信息，并将图片保存到训练数据目录
def gen_train_data(data_dir):
    file_lists = os.listdir(os.path.sep.join([data_dir, 'use']))
    # x <= 86
    file_lists = [x for x in file_lists if int(x) in[10001,10002, 10003,10004,] ]
    for fitem in file_lists:
        print('开始处理文件 -->：', fitem)
        gen_data_step(data_dir, fitem)

def gen_data_step(data_dir, file_name):
    # global GEN_FILE_IDX
    q_name = os.path.sep.join([data_dir, 'use', file_name, '{}_qs.json'.format(file_name)])
    img_name = os.path.sep.join([data_dir, 'use', file_name, '{}_img.json'.format(file_name)])

    with open(q_name, 'r', encoding='utf-8') as f:
        q_map = json.load(f)

    with open(img_name, 'r', encoding='utf-8') as f:
        q_img_map = json.load(f)

    q_contents = __get_question_contents__(q_map['question_list'])



    for epoch in range(0, 20):
        for idx, q_item in enumerate(q_contents):
            print('开始处理文件：', file_name, '批次', epoch ,' 第 ' , idx ,' 条问题!')
            tmp_dir = os.path.sep.join([data_dir, 'tmp'])
            try:
                random_flag = False if epoch == 0 else True
                # random_flag = True
                # 清空生成PDF目录下面的所有文件
                for item in os.listdir(tmp_dir):
                    os.remove(os.path.sep.join([tmp_dir,item]))
                # 根据内容生成 latex 内容

                latexs, latex_color = __gen_content_latex__(data_dir, file_name, q_item, q_img_map, random_flag=random_flag)
                if len(latexs) > 0:
                    # 生成PDF文件
                    __gen_content_pdf__(data_dir, latexs, False)
                    __gen_content_pdf__(data_dir, latex_color, True)
                    image,formula_pos, pic_pos = gen_latex_img_pos(data_dir, imgH=1860)
                    np.savetxt(os.path.sep.join([data_dir,'train','anno', f'{file_name}_{epoch}_{idx}.ppic']),np.array(pic_pos),'%.3f', ',', )
                    np.savetxt(os.path.sep.join([data_dir,'train','anno', f'{file_name}_{epoch}_{idx}.pmath']),np.array(formula_pos),'%.3f', ',', )
                    cv2.imwrite(os.path.sep.join([data_dir,'train','images', f'{file_name}_{epoch}_{idx}.png']), image)
                    # GEN_FILE_IDX = GEN_FILE_IDX + 1
            except Exception as e:
                print(e)

# 根据PDF得到数学公式、试卷图片及其位置定位信息
def __get_image_and_pos__(data_dir):
    pass


# 根据latex生成PDF文件
def __gen_content_pdf__(data_dir, latexs, is_color=False):
    if is_color:
        file_name = os.path.sep.join([data_dir, 'tmp','tmp_color'])
    else:
        file_name = os.path.sep.join([data_dir, 'tmp','tmp'])
    if len(latexs) == 0:
        return

    doc = Document()
    doc.packages.add(Package('ctex'))
    doc.packages.add(Package('color'))
    doc.packages.add(Package('xcolor'))  
    doc.packages.add(Package('graphicx'))  
    doc.packages.add(Package('geometry'))  
    doc.packages.add(Package('textcomp'))
    doc.packages.add(Package('caption','format=hang,font=small,textfont=it'))
    doc.append(NoEscape(r'\newgeometry{left=1cm,bottom=1cm}'))

    doc.append(NoEscape(r'\begin{figure}[ht]'))
    doc.append(NoEscape(r'开始'))
    doc.append(NoEscape(r'\end{figure}'))

    doc.append(NoEscape(r'\begin{figure}[ht]'))

    for item in latexs:
        doc.append(NoEscape(item))
        doc.append('\n')
    doc.append(NoEscape(r'\end{figure}'))

    doc.append(NoEscape(r'\begin{figure}[ht]'))
    # 结束标记靠右
    doc.append(NoEscape(r'\raggedleft 结束'))
    doc.append(NoEscape(r'\end{figure}'))


    doc.generate_pdf(file_name,clean_tex=True,compiler='xelatex')


# 根据文本内容生成latex格式
# random flag 是否随机替换图片
def __gen_content_latex__(data_dir, file_name, content, image_map, has_color=False, random_flag=False):
    latex_lists = []
    later_color_lists = []

    # 检测image map中的图片是否存在
    image_keys_exists = []
    for key in image_map.keys():
        img_name, width, height = image_map[key]
        if os.path.exists(os.path.sep.join([data_dir,'use', file_name, img_name])):
            image_keys_exists.append(key)
    
    content_lists = content.split('\n')
    for citem in content_lists:
        normal_latex = citem
        color_latex  = citem
        img_flags = re.findall(r'\{img:\d+\}', citem)

        for flag in img_flags:
            img_idx = flag.replace('}','').split(':')[1]

            if random_flag:
                # 选择图片高度、宽度类似的
                o_img_name, o_width, o_height = image_map[img_idx]
                if int(o_width) != 0:
                    random.shuffle(image_keys_exists)
                    for ikey in image_keys_exists:
                        # ikey = str(ikey)
                        n_img_name, n_width, n_height = image_map[ikey]
                        if int(n_width) / int(o_width) < 1.2:
                            img_idx = ikey
                            break

            img_name, width, height = image_map[img_idx]
            width = int(width)
            height = int(height)
            if img_name.find('.gif') == -1:
                radio = np.random.uniform(0.75, 1.0)
                width = int(width * radio)
                height = int(height * radio)
            # 检测图片是否存在：
            if width != 0 or height != 0:
                if os.path.exists(os.path.sep.join([data_dir,'use', file_name, img_name])):
                    if img_name.find('.gif') != -1:
                        # 复制图片到PDF生成目录， 并将gif转换成png
                        im = Image.open(os.path.sep.join([data_dir,'use', file_name, img_name]))
                        transparency = im.info['transparency'] 
                        im.save(os.path.sep.join([data_dir,'tmp', '{}.png'.format(img_name.split('.')[0])]), transparency=transparency)
                        color_latex = color_latex.replace(flag, r'\fcolorbox{red}{red} {\raisebox{-0.3\height}{\makebox[%spt]{\strut{\includegraphics[width=%spt, height=%spt]{%s}}}}}' % (width, width, height, '{}.png'.format(img_name.split('.')[0])))
                        normal_latex = normal_latex.replace(flag, r'\fcolorbox{white}{white} {\raisebox{-0.3\height}{\makebox[%spt]{\strut{\includegraphics[width=%spt, height=%spt]{%s}}}}}' % (width, width, height, '{}.png'.format(img_name.split('.')[0])))
                    else:
                        shutil.copy(os.path.sep.join([data_dir,'use',file_name, img_name]), os.path.sep.join([data_dir,'tmp',img_name]))
                        color_latex = color_latex.replace(flag, r'\fcolorbox{blue}{blue} {\raisebox{-0.3\height}{\makebox[%spt]{\strut{\includegraphics[width=%spt, height=%spt]{%s}}}}}' % (width, width, height, img_name))
                        normal_latex = normal_latex.replace(flag, r'\fcolorbox{white}{white} {\raisebox{-0.3\height}{\makebox[%spt]{\strut{\includegraphics[width=%spt, height=%spt]{%s}}}}}' % (width, width, height, img_name))
                else:
                    normal_latex = normal_latex.replace(flag, '')          
                    color_latex = color_latex.replace(flag,'')
            else:
                normal_latex = normal_latex.replace(flag, '')          
                color_latex = color_latex.replace(flag,'')

        latex_lists.append(normal_latex)
        later_color_lists.append(color_latex)
    # print('latex lists:', latex_lists)
    return latex_lists, later_color_lists


# 读取JSON文件，得到问题列表
def __get_question_contents__(q_lists):
    # 选择parent_id 不为0， 且为二级答题题目的问题
    q_sel_lists = [x['question'] for x in q_lists if len(x['question']['qid'].split('_')) == 3]
    # # print(q_lists[0]['question']['qid'])
    # print(q_idx_lists)
    c_lists = []
    for item in q_sel_lists:
        sub_q_lists = [x['question'] for x in q_lists if x['question']['pqid'] == item['qid']]
        # print(sub_q_lists)
        # content = ''
        if len(sub_q_lists) > 0:
            content = '\n'.join([x['content'] for x in sub_q_lists if len(x['content'].strip()) > 1])
            content = item['content'].strip() + '\n' + content
        else:
            content = item['content'].strip()

        c_lists.append(content)
    # print(c_lists)
    return c_lists


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

# 生成数据增加到lmdb数据库里
def add_data_lmdb(data_dir):
    outputPath = os.path.sep.join([data_dir, 'lmdb'])
    env = lmdb.open(outputPath, map_size=500511627)  

    # env = lmdb.open(outputPath, map_size=500511627) 
    nSamples = 0
    try: 
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('total'.encode()))
    except Exception as e:
        pass


    key_idx = nSamples

    # key_idx = 5682

    file_lists = os.listdir(os.path.sep.join([data_dir, 'train','images']))
    for idx, fitem in enumerate(file_lists):
        file_name = fitem.split('.')[0]

        with open(os.path.sep.join([data_dir, 'train','images',fitem]),'rb') as f:
            image_data = f.read()

        # print('------------------------------------')
        # print('file_name:', fitem)
        # print('math pos name:', os.path.sep.join([data_dir,'train','anno', f'{file_name}.pmath']))

        pic_pos = np.loadtxt(os.path.sep.join([data_dir,'train','anno', f'{file_name}.ppic']), dtype=np.float, delimiter=',')
        math_pos = np.loadtxt(os.path.sep.join([data_dir,'train','anno', f'{file_name}.pmath']), dtype=np.float,delimiter=',')
        # print('math pos:', math_pos)
        all_pos_array = np.vstack((pic_pos.reshape(-1,5), math_pos.reshape(-1,5)))
        # np.array 转成bytes, 恢复过来采用y = np.frombuffer(k, dtype=i.dtype), y.reshape(-1,5)
        # print('pos array:', all_pos_array.astype(np.uint8))
        pos_bytes = all_pos_array.tobytes(order='C')

        cache = {}
        key_idx = key_idx + 1

        cache[f'pos_{key_idx}'] = pos_bytes
        cache[f'img_{key_idx}'] = image_data
        cache['total'] = str(key_idx).encode()
        
        writeCache(env, cache)

    print('增加到数据缓存完成 !')

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

# 取已切分好的数据进行训练
def add_clip_data_lmdb(data_dir, file_number=None):

    def __get_image__(img_file):
        print('file name :', img_file)
        if img_file.find('.gif') != -1:
            im = Image.open(img_file)
            print('im:', im)
            # frames = list(iter_frames(im))
            transparency = im.info['transparency'] 
            # print('im:', transparency)
            png_buf = six.BytesIO()
            im.save(png_buf, format='PNG', transparency=transparency)
            image_bin = png_buf.getvalue()
            image = cv2.imdecode(np.frombuffer(image_bin, np.uint8),cv2.IMREAD_UNCHANGED)
            trans_mask = image[:,:,3] == 0
            #replace areas of transparency with white and not transparent
            image[trans_mask] = [255, 255, 255, 255]

            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            print('image shape:', image.shape)
            # cv2.imwrite('d:\\test.jpg', image)
            # print('image:', image)
            # plt.imshow(image)
            # plt.show()
        else:
            image = cv2.imread(img_file)
        _, image_data = cv2.imencode(".png", image)
        return image.shape,image_data.tobytes()         

    def __write_to_lmdb__(env, key_idx, img_file, type='formula'):
        image_shape, image_data = __get_image__(img_file)
        height, width, _ = image_shape
        cache = {}
        cache[f'img_{key_idx}'] = image_data
        if type == 'formula':
            pos = np.array([[0,0,width, height, 0]],dtype=np.float)
        else:
            pos = np.array([[0,0,width, height, 1]], dtype=np.float)


        cache[f'pos_{key_idx}'] = pos.tobytes(order='C')
        cache['total'] = str(key_idx).encode()
        writeCache(env, cache)
        print('write to lmdb :', img_file, ' pos:', pos, ' shape:', pos.shape)

    outputPath = os.path.sep.join([data_dir, 'lmdb'])
    env = lmdb.open(outputPath, map_size=500511627)  
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('total'.encode()))

    key_idx = nSamples + 1
    print('begin key idx :', key_idx)

    file_lists = os.listdir(os.path.sep.join([data_dir, 'use']))
    if file_number is None:
        file_lists = [x for x in file_lists if   int(x) <= 70]  # int(x) > 50 and
    else:
        file_lists = [x for x in file_lists if int(x) in file_number]
    for fitem in file_lists:
        print('开始处理文件 -->：', fitem)
        sub_file_lists = os.listdir(join(data_dir, 'use', fitem))
        math_file_lists = [join(data_dir, 'use',fitem, x) for x in sub_file_lists if x.find('.gif') != -1]
        pic_file_lists = [join(data_dir, 'use',fitem, x) for x in sub_file_lists if (x.find('.png')!=-1 or x.find('.jpg') != -1)]
        if file_number is None:
            random_sel = np.random.randint(0, len(math_file_lists), int(len(math_file_lists)*0.2)).tolist()
        else:
            random_sel = np.random.randint(0, len(math_file_lists), int(len(math_file_lists))).tolist()

        print('random sel :', len(random_sel))
        for idx in random_sel:
            __write_to_lmdb__(env, key_idx, math_file_lists[idx],type='formula')    
            key_idx = key_idx + 1

        for pitem in pic_file_lists:
            __write_to_lmdb__(env, key_idx, pitem,type='pic') 
            key_idx = key_idx + 1

# def count_type(env):
#     with env.begin(write=False) as txn:
#         nSamples = int(txn.get('total'.encode()))


#     for dtype in [0,1]:
#         for idx in range(nSamples):

#             target =  np.frombuffer(txn.get(target_key.encode()), dtype=np.float)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备定位数据')
    parser.add_argument("--config_file", default="D:\\PROJECT_TW\\git\\myproject\\NLP\lib\\testpaper\\bootstrap.yml", help="配置文件路径", type=str)
    parser.add_argument("--source_dir", default="D:\\PROJECT_TW\\git\\data\\testpaper\\source\\paper", help="原始文件路径", type=str)
    parser.add_argument("--dest_dir", default="D:\\PROJECT_TW\\git\\data\\mathdetect\\source", help="生成后的目的地址", type=str)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file) 
    # prepare_train_data(args.source_dir, args.dest_dir)
    # prepare_train_images(args.dest_dir)

    # 调用NLP/lib/testpaper/main.py batch_handle 方法，生成试卷的JSON内容

    # 生成图片位置坐标信息，并保存到训练数据目录 
    # gen_train_data(args.dest_dir)

    # 将图片和坐标信息保存到缓存数据库
    # add_data_lmdb(args.dest_dir)


    # 将已分割的子图片加入到缓存数据库存
    add_clip_data_lmdb(args.dest_dir, file_number=[1000,])
    # add_clip_data_lmdb(args.dest_dir, file_number=None)
