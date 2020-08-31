import argparse
from torchvision import transforms
from os.path import join
from models import SiameseNetwork
import torch
import cv2
import torch.nn.functional as F

transform = transforms.ToTensor()

def test_encode(model, data_dir, file_name):
    image = cv2.imread(join(data_dir,file_name), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image.copy(), (400,400), interpolation=cv2.INTER_AREA)
    image = transform(image)
    image = image.unsqueeze(0)
    # print('input image:', image.size())
    image_embed = model.embed_image(image)
    return image_embed


def test_distance(model, data_dir, file_name_1, file_name_2):
    embed_1 = test_encode(model, data_dir, file_name_1)
    embed_2 = test_encode(model, data_dir, file_name_2)
    distance = F.pairwise_distance(embed_1, embed_2, keepdim = True)
    print('distance :', distance)

    # print('image embed :',  image_embed.size())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path',default=r'D:\PROJECT_TW\git\data\siamese\weights\siame_best_net.pth', type=str, help='path of the evaluated model')
    parser.add_argument('--file_name',default='0_html_61e0f461ff4d8c33.png', type=str, help='path of the evaluated model')
    parser.add_argument('--dest_file_name',default='21_html_test.png', type=str, help='path of the evaluated model')
    parser.add_argument('--data_dir',default=r'D:\PROJECT_TW\git\data\siamese\images', type=str, help='path of the evaluated model')
    args = parser.parse_args()    
# 21_html_test.png
    model = SiameseNetwork()
    model.load_state_dict(torch.load(args.model_path,map_location=torch.device('cpu')))

    print('model :', model)
    model.eval()
    # test_encode(model, args.data_dir, args.file_name)
    test_distance(model, args.data_dir, args.file_name, args.dest_file_name)   