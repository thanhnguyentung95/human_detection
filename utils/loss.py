import torch


def compute_loss(preds, targets, model):
    device = targets.device
    lbox, lcls, lobj = torch.zeros(1, device=device)
    tbox, tcls, tanchors, tindices = build_targets(preds, targets, model)
    
    BCEcls = nn.BCEWithLogitsLoss.to(device)
    BCEobj = nn.BCEWithLogitsLoss.to(device)
    
    for i, pi in enumerate(preds): # each YOLOLayer output
        b, a, gj, gi = tindices[i]  # batch(image), anchor, grid j, grid i
        
        ps = pi[b, a, gj, gi] # prediction set
        
        n = b.shape[0] # number of targets
        
        tobj = torch.zeros_like(pi[..., 0]).to(device)
        
        if n > 0:
            # calculate regression loss
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * tanchors[i] 
            pbox = torch.cat((pxy, pwh), 1)
            giou = bbox_iou(pbox, tbox[i])
            lbox += (1 - giou).mean()
            
            # calculate classification loss
            if hyp.nc > 1: # more than one class
                t = torch.full_like(ps[:, 5:], 0, device=device) # intialize with full zeros
                t[range(n), tcls] = 1 # assign target class
                lcls += BCEcls(ps[:, 5:], t)
                
            # construct objectness
            tobj[b, a, gj, gi] = giou.detach().clamp(0).type(tobj.dtype)
            
        lobj += BCEobj(pi[:, 4], tobj)
        
    bs = p.shape[0] # batch size
    loss = lbox + lcls + lobj
    return loss * bs, torch.cat((lobj, lcls, lbox, loss)).detach()
    
    
def build_targets(preds, targets, model):
    device = targets.device
    tbox, tcls, tanchors, indices = [], [], [], [] 
    gain = torch.ones(6, device=device)
    nt = len(targets) # number of targets 
    off = torch.tensor([0, 1], [0, -1], [1, 0], [-1 0])
    
    for i, anchors in enumerate(model.anchors):
        a, t, offsets = [], targets, 0
        if nt > 0:
            # filter out non responsible anchors
            na = len(anchors) # number of anchors
            gain[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            t = t * gain
            a = torch.tensor(range(na)).view(na, 1).repeat(1, nt)
            r = t[None, :, 4:6] / anchors[:, None] # ratio
            m = torch.max(r, 1/r).max(2)[0] < model.hyp['anchor_t'] # mask for filter out non responsible anchors
            a, t = a[m], t.repeat(na, 1, 1)[m] # filter out
            
            # requires nearby grids also responsible for detecting object
            gx = t[:, 2]
            gy = t[:, 3]
            up = (hy % 1 < 0.5) & (hy > 1) # objects on the upper part of grid
            lo = (hy % 1 > 0.5) & (hy < gain[2] - 1) # objects on the lower part of grid
            le = (hx % 1 < 0.5) & (hx > 1) # object on the left of grid
            ri = (hx % 1 > 0.5 ) * (hx < gain[3] - 1) # object on the right of grid
            
            # add additional targets
            t = torch.cat((t, t[up], t[lo], t[le], t[ri]), 0)
            a = torch.cat((a, a[up], a[lo], a[le], a[ri]), 0)
            
            # offset for additional targets
            z = torch.zeros_like(t[:, :2])
            offsets = torch.cat((z, z[up] + off[0], z[lo] + off[1], z[le] + off[2], z[ri] + off[3]), 0)
        
        b, c = t[:, :2].T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]        
        gij = (gxy - offsets).long()
        gi, gj = gij.T
        
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        tcls.append(c)
        tanchors.append(anchors[a])
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        
        return tbox, tcls, tanchors, indices
        

def bbox_iou(box1, box2, CIoU=True): # xywh format
    '''
        CIoU: https://arxiv.org/pdf/1911.08287.pdf
    '''
    box1 = box1.T
    box2 = box2.T
    
    # calculate IoU
    
    b1_x1, b1_x2, b1_y1, b1_y2 = box1[0] - box1[2]/2, box1[0] + box1[2]/2, box1[1] - box1[3], box1[1] + box1[3]
    b2_x1, b2_x2, b2_y1, b2_y2 = box2[0] - box2[2]/2, box2[0] + box2[2]/2, box2[1] - box2[3], box2[1] + box2[3]
    
    inter = (torch.min(b1_x2, b2_x2)  - torch.max(b1_x1, b2_x1)).clamp(0) *   \
            (torch.min(b1_y2, b2_y2)  - torch.max(b1_y1, b2_y1)).clamp(0) # intersection area
            
    union = (box1[2] * box1[3] + 1e-16) + (box2[2] * box2[3]) - inter
    
    iou = inter/union
    
    if CIoU:
        # square of diagonal length of smallest enclosing box covering two boxes.
        c = (torch.max(b1_x2, b2_x2)  - torch.min(b1_x1, b2_x1)) ** 2 + \
            (torch.max(b1_y2, b2_y2)  - torch.min(b1_y1, b2_y1)) ** 2 + 1e-16
            
        # square of distance between centers of two boxes
        p = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
        
        v = (4/ math.pi ** 2) * torch.pow(torch.atan(box1[2]/box1[3]) + torch.atan(box2[2]/box2[3]), 2)
        
        with torch.no_grad():
            alpha = v / ((1- iou) + v + 1e-16)
        
        return iou - (p/c + alpha*v)
    return iou
            
            
        
        
        
        
        