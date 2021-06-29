# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps





class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        pred = pred
        true = true
        loss = self.loss_fcn(pred, pred)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(epoch,p, targets, model):  # predictions, targets, model
    device = targets.device
    

    loss_anchor = compute_loss_free_anchor(p[3],targets,model)
    # if loss_anchor !=0:
    # loss_anchor *=0.5
    bs = p[0][...,0].shape[0]
    
    # # print(bs)
    if epoch<300 and loss_anchor!=0:
        return loss_anchor*bs, torch.cat((loss_anchor, loss_anchor, loss_anchor, loss_anchor)).detach()
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        if i ==3:
            continue
        # pi = pi.repeat()
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        a = torch.zeros(a.shape,device=device).to(torch.int64)
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        n = b.shape[0]  # number of targets
        # print(n)
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets


            # Regression
            
            # print(ps.shape)
            if True:
                
                panchor = (p[3][i].sigmoid()*2)**3*2
                panchor = panchor[b,gj,gi]
                # print(panchor.shape)
                # skskskskks = ps[:, 2:4]
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = ps[:, 2:4].sigmoid()* panchor
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, DIoU=True,CIoU=True)  # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()
            # else:
            #     pxy = ps[:, :2].sigmoid() * 2. - 0.5
            #     pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            #     pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            #     iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            #     lbox += (1.0 - iou).mean()*2  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size
    # print(loss_anchor)

    # loss = lbox + lobj + lcls
    # if loss_anchor!=0:
    #     loss = lbox+lobj+lcls+loss_anchor
    #     loss =loss*1
    #     return loss * bs, torch.cat((lbox, lobj, loss_anchor, loss)).detach()
    # else:
    loss = lbox +lobj+lcls
        # loss = loss*3
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    na = 3
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        if i==0:
            anchors = det.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            # print(t[...,4:6].shape)
            j =  torch.max(t[...,4:6],2)[0] < 16
            # j = j.repeat()
            # print(j.shape)
            t = t[j]  # filter 
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j),))
            t = t.repeat((off.shape[0], 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            offsets = 0
        
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy

            gxy_ = torch.round(gxy)

            gwh = t[:, 4:6]  # grid wh

            gij = (gxy_ - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        if i==1:
            anchors = det.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            # print(t[...,4:6].shape)
            # j =  torch.max(t[...,4:6],2)[0] < 16
            j =  torch.max(t[...,4:6],2)[0] >4
            j1_ = torch.max(t[...,4:6],2)[0]<12
            j = j & j1_
            # j = j.repeat()
            # print(j.shape)
            t = t[j]  # filter 
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j),))
            t = t.repeat((off.shape[0], 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            offsets = 0
        
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy

            
            gxy_ = torch.round(gxy)

            gwh = t[:, 4:6]  # grid wh

            gij = (gxy_ - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        if i==2:
            anchors = det.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            # print(t[...,4:6].shape)
            j =  torch.max(t[...,4:6],2)[0] >4
            # j = j.repeat()
            # print(j.shape)
            t = t[j]  # filter 
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j),))
            t = t.repeat((off.shape[0], 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            offsets = 0
        
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy

            
            gxy_ = torch.round(gxy)

            gwh = t[:, 4:6]  # grid wh

            gij = (gxy_ - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        else:
            anchors = det.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                
                j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # print(j.shape)
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j),))
                t = t.repeat((off.shape[0], 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            # else:
            #     t = targets[0]
            #     offsets = 0

            #     # Define
            # b, c = t[:, :2].long().T  # image, class
            # gxy = t[:, 2:4]  # grid xy
            # gwh = t[:, 4:6]  # grid wh
            # gij = (gxy - offsets).long()
            # gi, gj = gij.T  # grid xy indices

            # # Append
            # a = t[:, 6].long()  # anchor indices
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # anch.append(anchors[a])  # anchors
            # tcls.append(c)  # class

    return tcls, tbox, indices, anch



def compute_loss_free_anchor(p, targets, model):  # predictions, targets, model
    device = targets.device
    targets_size =targets.shape
    lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, indices, anchors = build_targets_freeanchor(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5])).to(device)
    # Focal loss
    g = 5
    
    if g > 0:
        BCEobj = FocalLoss(BCEobj, g)

    # Losses
    no = len(p)  # number of outputs

    for i, pi in enumerate(p):  # layer index, layer predictions

        b, gj, gi = indices[i]  # image, gridy, gridx

        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        tobj_shape = tobj.shape
        # print(tobj.shape)

        n = b.shape[0]  # number of targets
        # print(n)

        if n:
            ps = pi[b, gj, gi]  # prediction subset corresponding to targets
            gxy = torch.cat((gj.view(gj.shape[0],-1),gi.view(gi.shape[0],-1)),1).to(device)
            pw = (ps[:, 0].sigmoid()*2)**2
            ph = (ps[:, 1].sigmoid()*2)**2
            pwh = (ps[:, :2].sigmoid()*2)**3*2

            an = anchors[i]
            aw = an[:,0]
            ah = an[:,1]
            lbox += anchor_loss(pwh,an).view(1)
        # else:
        #     return 0


    loss = lbox*0.02
    # if lobj
    return loss



def build_targets_freeanchor(p, targets, model):
    
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    # print(targets.shape)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    nt = targets.shape[0]  # number of anchors, targets
    tcls, indices, anch = [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    # ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias

    # print(p[0].shape)
    for i in range(len(p)):
        # anchors = det.anchors[i]
        # print(p[i].shape)
        gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]  # xyxy gain
        # print(gain)

        # Match targets to anchors
        if nt:
            t = targets * gain
            r = t[:, 4:6]  # wh ratio
    
        
            if i ==0:
                j =  torch.max(t[:,4:6],1)[0] < 16  # compare
            elif i==1:
                j =  torch.max(t[:,4:6],1)[0] >6
                j1_ = torch.max(t[:,4:6],1)[0]<12
                j = j & j1_
                
            else:
                j =  torch.max(t[:,4:6],1)[0] >6

            # print(j)
            # print(torch.max(t[:,4:6],0)[0] < 2)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter
            offsets = 0
            b, c = t[:,:2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gxy = torch.round(gxy)
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            indices.append((b, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, grid indices
            # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(gwh)  # anchors
            tcls.append(c)  # class

    return tcls, indices, anch


def anchor_loss(p,d):
    loss_func = torch.nn.MSELoss(reduce=None, size_average=True).to(p.device)
    loss = loss_func(p,d)
    # loss  = torch.mean(loss)
    # IOU loss function:



    # print((p-d)**2)
    # print(loss)
    return loss


