import cv2
import torch
from utils.data import create_dataloader
from models.model import Yolov4Tiny


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hard code, path to list image file
train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

# create data loader
dataloader = create_dataloader(train_path, batch_size=8, img_size=416)

# create model
model = Yolov4Tiny().to(device)

# get img, label and write to out.jpg for checking
img, label = next(iter(dataloader))

# pass data through model
output = model(img.to(device=device, non_blocking=True).float())

print('output', output)
print('shape', output.shape)
