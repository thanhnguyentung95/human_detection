import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, path, img_size, cache_labels=False):
        # Assert if file exist
        assert os.path.isfile(path), path + 'does not exist'
        
        self.img_size = img_size

        # Get path dataset folder
        dirname = os.path.dirname(path)

        # Load image paths
        with open(path) as f:
            self.img_paths = f.readlines()
            self.img_paths = [p.replace('./', dirname + os.sep).strip() for p in self.img_paths]

        self.len = len(self.img_paths)
        
        # Get corresponding labels for images
        img_folder = os.sep + 'images' + os.sep
        label_folder = os.sep + 'labels' + os.sep
        self.label_paths = [img_path.replace('.jpg', '.txt').replace(img_folder, label_folder) 
                             for img_path in self.img_paths]
        if cache_labels:
            from tqdm import tqdm
            
            self.labels = []
            self.shapes = []
            pbar = tqdm(zip(self.label_paths, self.img_paths))
            for label_path, img_path in pbar:
                if not os.path.isfile(label_path): # label not exist
                    continue
                img = Image.open(img_path)
                shape = img.size
                with open(label_path, 'r') as f:
                    label = np.array([line.split() for line in f.read().splitlines()], dtype=np.float32)
                if len(label):
                    for l in label:
                        self.labels.append(l)
                        self.shapes.append(shape)            
            self.labels = np.array(self.labels)
            self.shapes = np.array(self.shapes)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, label = None, None
        
        # Get image and its label by index
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]
        
        img = imread(img_path, self.img_size)
        
        label = []
        if os.path.isfile(label_path):
            with open(label_path) as f:
                label = np.array([line.split() for line in f.read().splitlines()], dtype=np.float32)
                
        if len(label) == 0:
            label = np.zeros((1, 5), dtype=np.float32)
        
        label = torch.from_numpy(label)
        # print('after convert label shape is: ', label.size())
        
        return img, label
    

def imread(img_path, img_size):
    img = cv2.imread(img_path)
    assert img is not None, 'Can not read ' + img_path
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1) # convert from channel last to channel first
    
    return img


def collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = list(labels)
    
    # horizontally stack index of labels
    for i, label in enumerate(labels):
        nl = label.shape[0] # number of labels (objects)
        index = torch.ones((nl, 1)) * i
        labels[i] = torch.hstack((index, label))
    
    imgs = torch.stack(imgs, 0)
    labels = torch.cat(labels, 0)
    
    return imgs, labels


def create_dataloader(path, img_size=416, batch_size=8):
    # representing a dataset.
    dataset = Data(path, img_size)
    
    # multi-process data loading
    nw = min(os.cpu_count(), batch_size)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=nw,
                                sampler=None,
                                collate_fn=collate_fn)
    
    return dataloader