import cv2
import torch
from utils.data import create_dataloader
from models.model import Yolov4Tiny

# hard code, path to list image file
train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create data loader
dataloader = create_dataloader(train_path, batch_size=8, img_size=416)

# create model
model = Yolov4Tiny().to(device)

# create optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.937, nesterov=True)

