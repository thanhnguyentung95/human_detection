import argparse
import cv2
from models.model import Yolov4Tiny
from utils.data import imread
import torch

def test(args):
    model = Yolov4Tiny().to('cuda')
    
    model.load_state_dict(torch.load(args.ckpt)['model'])
    model.eval()
    
    img = imread(args.img, args.img_size)
    img = torch.tensor(img)[None, ...].to('cuda')
    
    preds, out = model(img.float())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='image for testing')
    parser.add_argument('--ckpt', type=str, required=True, help='model checkpoint for testing')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters')
    parser.add_argument('--img_size', type=int, default=416, help='image size')
    args = parser.parse_args()
    
    test(args)