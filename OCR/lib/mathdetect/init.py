import argparse
from data import *

def init_args(params=None):
    '''
    Read arguments and initialize directories
    :return: args
    '''

    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--root_path', default='D:\\PROJECT_TW\\git\\data\\mathdetect', 
                        type=str, help='data root path')
    parser.add_argument('--dataset', default='GTDB', choices=['GTDB'],
                        type=str, help='choose GTDB')
    parser.add_argument('--dataset_root', default='D:\\PROJECT_TW\\git\\data\\mathdetect\\source',
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=5, type=int,
                        help='Batch size for training')
    # parser.add_argument('--resume', default='D:\\PROJECT_TW\\git\\data\\mathdetect\\ckpts\\weights_math_detector\\best_ssd512.pth', type=str,
    parser.add_argument('--resume', default=None, type=str,    
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in data loading')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Learning Rate")
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. "
                        "When sample method is Exponential deca;"
                        "Or a constant in Inverse sigmoid decay Equation. "
                        "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )

    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Learning Rate Decay Patience")

    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Alpha for the multibox loss')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--exp_name', default='math_detector',  # changed to exp_name from --save_folder
                        help='It is the name of the experiment. Weights are saved in the directory with same name.')
    parser.add_argument('--layers_to_freeze', default=20, type=float,
                        help='Number of VGG16 layers to freeze')
    parser.add_argument('--model_type', default=300, type=int,
                        help='Type of ssd model, ssd300 or ssd512')
    parser.add_argument('--suffix', default="_10", type=str,
                        help='Stride % used while generating images or dpi from which images was generated or some other identifier')
    parser.add_argument('--training_data', default="training_data", type=str,
                        help='Training data to use. This is list of file names, one per line')
    parser.add_argument('--validation_data', default="valid_data", type=str,
                        help='Validation data to use. This is list of file names, one per line')
    parser.add_argument('--use_char_info', default=False, type=bool,
                        help='Whether to use char position info and labels')
    parser.add_argument('--cfg', default="ssd300", type=str,
                        help='Type of network: either gtdb or math_gtdb_512')
    parser.add_argument('--loss_fun', default="fl", type=str,
                        help='Type of loss: either fl (focal loss) or ce (cross entropy)')
    # parser.add_argument('--kernel', default="3 3", type=int, nargs='+',
    #                     help='Kernel size for feature layers: 3 3 or 1 5')
    # parser.add_argument('--padding', default="1 1", type=int, nargs='+',
    #                     help='Padding for feature layers: 1 1 or 0 2')
    parser.add_argument('--neg_mining', default=False, type=bool,
                        help='Whether or not to use hard negative mining with ratio 1:3')
    parser.add_argument('--log_dir', default="D:\\PROJECT_TW\\git\\data\\mathdetect\\log", type=str,
                        help='dir to save the logs')
    parser.add_argument('--stride', default=0.1, type=float,
                        help='Stride to use for sliding window')
    parser.add_argument('--window', default=1200, type=int,
                        help='Sliding window size')
    parser.add_argument('--detect_type', default='formula', type=str,help='pic or formula')

    parser.add_argument('--pos_thresh', default=0.5, type=float,
                        help='All default boxes with iou>pos_thresh are considered as positive examples')

    if params is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(params)

    args.kernel = (1,5)
    args.padding = (0,2)

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logging.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                            "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists("weights_" + args.exp_name):
        os.mkdir("weights_" + args.exp_name)

    return args
