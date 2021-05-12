import cv2
import math
import torch
import numpy as np
from utils.data import create_dataloader
from models.model import Yolov4Tiny
from torch.optim import lr_scheduler
from utils.loss import compute_loss
from torch import nn
import matplotlib.pyplot as plt

# HARD CODE area
# path to list image file
train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'
lr = 1e-2
momentum = 0.937
batch_size = 8
img_size=416
epochs = 20

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create data loader, dataset information
dataloader = create_dataloader(train_path, batch_size=batch_size, img_size=img_size)
nb = len(dataloader.dataset)   # number of batches 

# create model
model = Yolov4Tiny().to(device)

# hyperparameter
hyp = {}
hyp['nc'] = 80
hyp['anchor_t'] = 4.0
model.hyp = hyp

# Gradient descent
# create optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

# learning rate scheduling
lf = lambda epoch: ((1 + math.cos(epoch * math.pi / epochs))/2) * 0.99 + 0.01
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# warm-up
wue = 3 # warm-up epochs
wub = wue * nb # warm-up batches
wu_batchs = [0, wub] # warm-up range (in batch)
max_lr = lf(wue) * lr
wu_lr = [0.01 * lr, max_lr] # warm-up range (in learning rate)

# systhesis large batch size
n_batch_size = 64
accumulate = n_batch_size / batch_size

# train
for epoch in range(epochs):
    model.train()
    # start epoch
    for i, (imgs, labels) in enumerate(dataloader):
        # start batch
        batch = i + epoch * nb
        
        # override learning rate if in warm up phrase
        if batch < wub:
            wulr = np.interp(batch, wu_batchs, wu_lr) # warm-up learning rate
            for g in optimizer.param_groups:
                g['lr'] = wulr
                                
        preds = model(imgs.float().to(device))
        
        loss, loss_item = compute_loss(preds, labels.to(device), model)                 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        
        # end batch
    scheduler.step()
    # end epoch
    
    