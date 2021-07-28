from utils.data import Data
from tqdm import tqdm
import yaml
import os
import numpy as np
from scipy.cluster.vq import kmeans


def calc_anchors(path='data/coco.yaml', n=6, img_size=416):
    
    def print_anchors(k):
        k = k[np.argsort(k.prod(axis=1))].round()
        r = wh0[:, None, :] / k[None, ...] # ratio
        print('r shape: ', r.shape)
        r = np.maximum(r, 1./r).max(axis=2).min(axis=1)
        recall = (r < 4).mean()
        print('recall: ',recall, ' - anchors: ', k)    
    
    k = [0] * n    
    # load data
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)    
    trainset = cfg['train']
    base = os.path.dirname(os. getcwd())
    trainset = trainset.replace('..', base)
    trainset = Data(trainset, img_size, cache_labels=True)    
    
    # get labels
    labels = trainset.labels
    wh = labels[:, 3:]
    shapes = trainset.shapes
    wh0 = img_size * wh * shapes / shapes.max(1, keepdims=True)
    
    # kmean
    s = wh0.std(0)
    k, _ = kmeans(wh0/s, n, iter=30)
    k *= s
    print_anchors(k)
    
    return k