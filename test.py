import argparse
import cv2
from models.model import Yolov4Tiny
from utils.data import imread
from utils.helper import non_max_suppression
import torch
import yaml

def test(args):
    device = 'cpu'
    with open(args.coco) as f:
        coco = yaml.load(f, Loader=yaml.FullLoader)
    classes = coco['names']
    
    model = Yolov4Tiny().to(device)
    
    model.load_state_dict(torch.load(args.ckpt)['model'])
    model.eval()
    
    img = imread(args.img, args.img_size)
    img = img[None, ...].to(device)
    
    preds, out = model(img.float())
    preds = non_max_suppression(preds.detach())

    img = cv2.imread(args.img)
    img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

    for pred in preds:
        for obj in pred:
            obj = obj.cpu()
            box = obj[:4].int().numpy()
            score = obj[4].item()
            c = obj[5].int().item()
            print(f'box: {box} - score: {score:>.3} - c: {classes[c]}')
            p1 = tuple(box[0:2])
            p2 = tuple(box[2:4])
            img = cv2.rectangle(img, p1, p2, (255, 128, 0), 2)
            
    cv2.imwrite('out' + args.img, img)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='image for testing')
    parser.add_argument('--ckpt', type=str, required=True, help='model checkpoint for testing')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters')
    parser.add_argument('--img_size', type=int, default=416, help='image size')
    parser.add_argument('--coco', type=str, default='data/coco.yaml', help='')
    args = parser.parse_args()
    
    test(args)