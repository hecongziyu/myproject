from entity.entity_main import EntityMaster
from entity.ner import NERecognition
from entity.align import EntityAlignment
from os.path import join


def test_main(data_dir):
    # text = '平行四边形，是在同一个二维平面内，由两组平行线段组成的闭合图形。平行四边形一般用图形名称加四个顶点依次命名。注：在用字母表示四边形时，一定要按顺时针或逆时针方向注明各顶点。相比之下，只有一对平行边的四边形是梯形。平行四边形的三维对应是平行六面体。'
    text = '平行四边形，是在同一个二维平面内，由两组平行线段组成的闭合图形。'    


    dict_file = join(data_dir, 'entity', 'dict_entity.txt')
    # pku_dict_file = join(data_dir, 'entity', 'pku_dict.txt')

    align = EntityAlignment()
    ner = NERecognition(dict_file=dict_file,pku_dict_file=pku_dict_file,seg_name='pkuseg', entity_aligment=align)
    

    em = EntityMaster(e_ner=ner, e_link=align, e_rel=None, k_graphy=None)

    # 注意对于text 以句号分隔成多个句子

    em.handle(text = text)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description="总体分析")
    args.add_argument("--dict_file", help="字典目录",default='D:\\PROJECT_TW\\git\\data\\finance\\dict.txt',type=str, required=False)
    args.add_argument("--data_dir", help="目录",default=r'D:\PROJECT_TW\git\data\kg',type=str, required=False)
    args = args.parse_args()
    test_main(data_dir = args.data_dir)