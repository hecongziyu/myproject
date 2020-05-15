import argparse
from build_vocab import Vocab, load_vocab
import torch
from decoding import LatexProducer
from latextransform import Resize,ConvertFromInts,BackGround
from model import Im2LatexModel
import cv2
from torchvision import transforms
from latexdataset import LatexDataset,collate_fn,MEANS,expand_width
from torch.utils.data import DataLoader
from latextransform import LatexImgTransform
from functools import partial
from matplotlib import pyplot as plt
import time


def load_model(model_path):
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model_args = checkpoint['args']    
    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model    

def valid(latex_producer,image_path, data_root):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resize = Resize(imgH=128)
    convertInts = ConvertFromInts()
    background = BackGround(data_root)
    image = resize(image)
    image = background(image)
    image = expand_width(image, imgH=128, max_width=600)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    print('input image size:', image_tensor.size())
    formula = latex_producer(image_tensor)
    plt.imshow(image)
    plt.show()
    return formula

def batch_valid(latex_producer,vocab,args):
    start = time.time()
    data_loader = DataLoader(
        LatexDataset(args, data_file=args.data_file,split='test', 
                     transform = None,
                     # transform=LatexImgTransform(imgH=128, mean=MEANS,data_root=args.dataset_root),
                     max_len=args.max_len),
                     batch_size=1,
                     # batch_size=args.batch_size,
                     # collate_fn=partial(collate_fn, vocab.sign2id),
                     pin_memory=True if use_cuda else False,
                     num_workers=4)
    print('data load time :', (time.time() - start ))
    for idx in range(5):
        start = time.time()
        # for imgs, tgt4training, tgt4cal_loss in data_loader:
        for data in data_loader:
            try:
                print(idx, ' load data time :',  (time.time() - start ))
                # start = time.time()
                # reference = latex_producer._idx2formulas(tgt4cal_loss)
                # results = latex_producer(imgs)
                # print('predict  time :',  (time.time() - start ))
                start = time.time()
                # print('results:', results)
                # print('tgt4cal_loss:', reference)
            except RuntimeError:
                break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path',default='D:\\PROJECT_TW\\git\\data\\im2latex\\ckpts\\best_ckpt.pt', type=str, help='path of the evaluated model')
    parser.add_argument('--image_file',default='D:\\PROJECT_TW\\git\\data\\im2latex\\gen_images\\19.png', type=str, help='path of the evaluated model')
    parser.add_argument('--dataset_root',default='D:\\PROJECT_TW\\git\\data\\im2latex', type=str, help='path of the evaluated model')
    parser.add_argument("--max_len", type=int, default=150, help="Max step of decoding")    
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020,help="The random seed for reproducing ")
    parser.add_argument('--data_file', default='latex_formul_normal.txt',help='data set root')    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    vocab = load_vocab(args.dataset_root)
    model = load_model(args.model_path)
    use_cuda = True if torch.cuda.is_available() else False

    print(model)

    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)    

    # batch_valid(latex_producer, vocab, args)

    formula = valid(latex_producer, args.image_file, args.dataset_root)
    print('formula:', formula)



