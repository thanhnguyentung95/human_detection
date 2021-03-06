import cv2
import os
import argparse
import math
import numpy as np
import yaml
from utils.data import create_dataloader
from utils.loss import compute_loss
from utils.helper import prepare_checkpoint_folder
from models.model import Yolov4Tiny
import torch
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(args):
    # path to list image file
    train_path = '/home/heligate/Documents/ScaledYOLOv4/coco/train2017.txt'

    # load hyper parameters
    with open('cfg/hyp.yaml', 'r') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    batch_size = args.batch_size
    img_size= args.img_size
    epochs = args.epochs

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create data loader, dataset information
    dataloader = create_dataloader(train_path, batch_size=batch_size, img_size=img_size)
    nb = len(dataloader)   # number of batches 

    # create model
    model = Yolov4Tiny().to(device)
    model.hyp = hyp

    # create optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True, weight_decay=hyp['weight_decay'])

    # learning rate scheduling
    lf = lambda epoch: ((1 + math.cos(epoch * math.pi / epochs))/2) * 0.99 + 0.01
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # warm-up
    lr = hyp['lr0'] # initial learning rate
    wue = 3 # warm-up epochs
    wub = wue * nb # warm-up batches
    wu_batchs = [0, wub] # warm-up range (in batch)
    max_lr = lf(wue) * lr
    wu_lr = [0.01 * lr, max_lr] # warm-up range (in learning rate)

    # systhesis large batch size
    n_batch_size = 32
    accumulate = n_batch_size / batch_size

    # resume
    start_epoch = 0
    exp = 0
    model_name = args.cfg.split('/')[1].replace('-', '_')
    ckpt_path = args.ckpt
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            exp = checkpoint['exp'] + 1
        except:
            pass
        print('scheduler: ', checkpoint['scheduler'])


    # train
    for epoch in range(start_epoch, epochs):
        # Tensorboard
        writer = SummaryWriter(f'runs/exp_{exp}')
        
        model.train()
        # start epoch
        progress = tqdm(dataloader, total=nb)
        title = '%10s' * 5 % ('Epoch', 'lobj', 'lcls', 'lbox', 'loss')        
        alobj, alcls, albox, atotal_loss = 0, 0, 0, 0   # loss accumulated
        print(title)
        for i, (imgs, labels) in enumerate(progress):
            # start batch
            batch = i + epoch * nb
            
            # override learning rate if in warm up phrase
            if batch < wub:
                wulr = np.interp(batch, wu_batchs, wu_lr) # warm-up learning rate
                for g in optimizer.param_groups:
                    g['lr'] = wulr
                                    
            preds = model(imgs.float().to(device))
            
            loss, loss_item = compute_loss(preds, labels.to(device), model)
            lobj, lcls, lbox, total_loss = loss_item
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lobj, lcls, lbox, total_loss = lobj.item(), lcls.item(), lbox.item(), total_loss.item()
            
            # set description
            desc = '%10.6s' * 5 % (epoch, lobj, lcls, lbox, total_loss)
            progress.set_description(desc)
            
            # accumulate loss
            alobj += lobj
            alcls += lcls
            albox += lbox
            atotal_loss += total_loss
    
            # end batch
        scheduler.step()
        # end epoch
        
        # average loss
        alobj /= nb
        alcls /= nb
        albox /= nb
        atotal_loss /= nb

        # create checkpoint
        checkpoint = {
            'exp': exp,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        
        writer.add_scalars('loss', {'obj': alobj,
                            'cls': alcls,
                            'box': albox,
                            'total': atotal_loss}, epoch)
        writer.close()
        
        # save state and log
        ckpt_folder = prepare_checkpoint_folder(exp)
        torch.save(checkpoint, ckpt_folder + os.sep + 'checkpoint_' + str(epoch) + '.pth')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov4-tiny.cfg', help='config of model')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint to resume')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters config')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--img_size', type=int, default=416, help='image size for training')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    args = parser.parse_args()
    
    train(args)
    
    
    