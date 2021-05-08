import torch

def compute_loss(preds, targets, hyp):
    device = targets.device
    lbox, lcls, lobj = torch.zeros(1, device=device)
    tbox, tcls, tanchors, tindices = build_targets(targets)
    
    BCEcls = nn.BCEWithLogitsLoss.to(device)
    BCEobj = nn.BCEWithLogitsLoss.to(device)
    
    for i, pi in enumerate(preds): # each YOLOLayer output
        b, a, gj, gi = tindices[i]  # batch(image), anchor, grid j, grid i
        
        ps = pi[b, a, gj, gi] # prediction set
        
        n = b.shape[0] # numbber of targets
        
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
    
    
def build_targets():
    pass
    
        
def bbox_iou(pbox, tbox):
    pass
            
            
        
        
        
        
        