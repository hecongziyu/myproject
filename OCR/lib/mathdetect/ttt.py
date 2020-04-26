from __future__ import division
import ssd as ssd
from data import *
import argparse
import importlib
importlib.reload(argparse)
importlib.reload(ssd)

def parse_args():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model', default='D:\\PROJECT_TW\\git\\data\\mathdetect\\weights\\AMATH512_e1GTDB.pth',type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.25, type=float,help='Final confidence threshold')
    parser.add_argument('--cuda', default=False, type=bool,help='Use cuda to train model')
    parser.add_argument('--dataset_root', default='../', help='Location of VOC root directory')
    parser.add_argument('--test_data', default="testing_data", help='testing data file')
    parser.add_argument('--verbose', default=False, type=bool, help='plot output')
    parser.add_argument('--suffix', default="_10", type=str, help='suffix of directory of images for testing')
    parser.add_argument('--exp_name', default="SSD", help='Name of the experiment. Will be used to generate output')
    parser.add_argument('--model_type', default=512, type=int, help='Type of ssd model, ssd300 or ssd512')
    parser.add_argument('--use_char_info', default=False, type=bool, help='Whether or not to use char info')
    parser.add_argument('--limit', default=-1, type=int, help='limit on number of test examples')
    parser.add_argument('--cfg', default="hboxes512", type=str,help='Type of network: either gtdb or math_gtdb_512')
    parser.add_argument('--batch_size', default=16, type=int,help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,help='Number of workers used in data loading')
#     parser.add_argument('--kernel', default="3 3",nargs='+', help='Kernel size for feature layers: 3 3 or 1 5')
#     parser.add_argument('--padding', default="1 1", nargs='+',help='Padding for feature layers: 1 1 or 0 2')
    parser.add_argument('--neg_mining', default=True, type=bool,help='Whether or not to use hard negative mining with ratio 1:3')
    parser.add_argument('--log_dir', default="logs", type=str,help='dir to save the logs')
    parser.add_argument('--stride', default=0.1, type=float,help='Stride to use for sliding window')
    parser.add_argument('--window', default=1200, type=int,help='Sliding window size')
    parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
    args = parser.parse_args([])
    return args

args = parse_args()
args.kernel = (1,5)
args.padding = (0,2)
gpu_id=0
num_classes=2
net = ssd.build_ssd(args, 'test', exp_cfg[args.cfg], gpu_id, args.model_type, num_classes)
# mod = torch.load(args.trained_model,map_location=torch.device('cpu'))
# mod_new = {k.replace('module.',''):v for k,v in mod.items()}
# net.load_state_dict(mod_new)

image_path = 'D:\\PROJECT_TW\\git\\data\\mathdetect\\images\\1_02.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
trans =BaseTransform(args.model_type,(104, 117, 123))
img = trans(image)
img = torch.from_numpy(img[0])
if args.cuda:
    img = img.cuda()
    net.to(0)
img.unsqueeze_(0)
img = img.permute(0,3,1,2)
net.eval()
# y, debug_boxes, debug_scores = net(img)
# net(img)
net(img)
