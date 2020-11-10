from os.path import join
from entity.nre.breds.breds import BREDS

def test_init_breds(data_root, sentence_file):
    pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="试卷导入功能")
    parser.add_argument("--data_root", default=r"D:\PROJECT_TW\git\data\kg\entity", help="模型参数、分词", type=str)
    parser.add_argument("--sentence_file", default=r"sentence_file.txt", help="模型参数、分词", type=str)    
    args = parser.parse_args()
