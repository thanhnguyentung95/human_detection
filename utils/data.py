import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, path, img_size):
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
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        print('index: ', index)
        img, label = None, None
        
        # Get image and its label by index
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]
        
        img = cv2.imread(img_path)
        assert img is not None, 'Can not read ' + img_path
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        with open(label_path) as f:
            label = np.array([line.split() for line in f.read().splitlines()], dtype=np.float32)
        
        # Convert
        label = torch.from_numpy(label)
        
        return img, label
    

def collate_fn(batch):
    img, label = zip(*batch)


def create_dataloader(path, img_size=640, batch_size=8):
    # representing a dataset.
    dataset = Data(path, img_size)
    
    # multi-process data loading
    nw = min(os.cpu_count(), batch_size)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=nw,
                                sampler=None,
                                collate_fn=None)
    
    return dataloader
    
    

    