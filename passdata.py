import cv2
from ..utils.data import create_dataloader
from ..models.model import Yolov4Tiny


# hard code, path to list image file
train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

# create data loader
dataloader = create_dataloader(train_path, batch_size=8, img_size=640)

# create model
model = Yolov4Tiny()

# get img, label and write to out.jpg for checking
img, label = next(data(dataloader))

# pass data through model
output = model(img)

print('output', output)
print('shape', output.shape)
