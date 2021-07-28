import os
import torch
import torchvision


def prepare_checkpoint_folder(exp):
    root_folder = 'runs' + os.sep + 'exp_' + str(exp)
    path = root_folder + os.sep + 'weights'
    # if not os.path.isdir(root_folder): # create if exp folder does not exist
    #     os.mkdir(root_folder)
    if not os.path.isdir(path): # create if exp folder does not exist
        os.mkdir(path)
    
    return path


def non_max_suppression(preds, conf_thres=0.1, iou_thres=0.4):
    nc = preds.shape[2] - 5
    max_wh = 4096
    
    outs = []
    for i, pred in enumerate(preds):
        box = pred[:, :4]
        box = xywh2xyxy(box)
        conf = pred[:, 4:5] # confidence
        c = pred[:, 5:] * conf  # class
        conf, c = c.max(1, keepdim=True)
        
        pred = torch.cat((box, conf, c.float()), 1)[conf.view(-1) > conf_thres]
        
        offset = pred[:, 5:6] * max_wh
        box, score = pred[:, :4] + offset, pred[:, 4]
        m = torchvision.ops.boxes.nms(box, score, iou_thres)
        outs.append(pred[m])
        
    return outs


def xywh2xyxy(box):
    nb = torch.full_like(box, 0)
    nb[:, 0] = box[:, 0] - box[:, 2]/2
    nb[:, 1] = box[:, 1] - box[:, 3]/2
    nb[:, 2] = box[:, 0] + box[:, 2]/2
    nb[:, 3] = box[:, 1] + box[:, 3]/2
    return nb