
import re
import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel

import torch.nn.functional as F

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
        loss = self.loss_fcn(pred, true)
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
def compute_loss_refinenet(p,targets,boxes,model):
    device = targets.device
    all_batch_target = build_targets_forbatch([320,320],targets,boxes)
    indices,tpoint,tcls,tbox = build_targets_forlayer(p, all_batch_target)
    
    bs = p[0][...,0].shape[0]
    
    # # print(bs)
    lcls, lbox, lobj,lpoint_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device),torch.zeros(1, device=device)
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
        b, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        n = b.shape[0]  # number of targets
        # print(n)
        if n:
            nt += n  # cumulative targets
            ps = pi[b, gj, gi]  # prediction subset corresponding to targets

            # Regression
            if True:
                p_fourpoint = (ps[:, :8].sigmoid()-0.5)*2
                lpoint_loss += F.mse_loss(p_fourpoint,tpoint[i])

                # print(pi.shape)
                pbox = get_rec_box_for_predict(p_fourpoint)

                # pxy = 
                # pwh = (ps[:, 2:4].sigmoid()*torch.tensor([2,2]).to(device)).to(device)

                # pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, DIoU=True,CIoU=True)  # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()

            # Objectness
            tobj[b, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
            # print(torch.max(tobj),torch.max(pi[..., 4]))

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s*0.5
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)*0.5
    lcls *= h['cls'] * s*0.5
    bs = tobj.shape[0]  # batch size
    # else:
    loss = lbox +lobj+lcls+lpoint_loss
    loss = loss*0.01
    return loss * bs, torch.cat((lbox, lobj, lcls,lpoint_loss, loss)).detach()



    
    # p:N*85*w*h
    # target:K*6
    # boxes: N*4
    return 0


def build_targets_forbatch(feature_size,target,bboxes):
    batch_size = len(bboxes)
    all_batch_target = []
    BOX_COUNT = 0

    for j in range(batch_size):
        im_target = target[target[:,0]==j]
        nt = im_target.shape[0]
        bbox = bboxes[j]
        if nt and len(bbox):
            im_target = im_target.detach()
            im_target[:,[2,4,6,8]] = im_target[:,[2,4,6,8]]*feature_size[0]
            im_target[:,[3,5,7,9]] = im_target[:,[3,5,7,9]]*feature_size[1]
            # im_target[:,4] = im_target[:,4]*feature_size[0]
            # im_target[:,5] = im_target[:,5]*feature_size[1]
            # im_target = im_target.repeat(len(bbox),1)
            im_target_res = torch.zeros_like(im_target).repeat(len(bbox),1)
            
            for i in range(len(bbox)):
                im_target_res[(i)*nt:(i+1)*nt,[2,4,6,8]] = (im_target[:,[2,4,6,8]]-bbox[i][0])/(bbox[i][2]-bbox[i][0])
                im_target_res[(i)*nt:(i+1)*nt,[3,5,7,9]] = (im_target[:,[3,5,7,9]]-bbox[i][1])/(bbox[i][3]-bbox[i][1])
                # im_target_res[(i)*nt:(i+1)*nt,4] = im_target[:,4]/(bbox[i][2]-bbox[i][0])
                # im_target_res[(i)*nt:(i+1)*nt,5] = im_target[:,5]/(bbox[i][3]-bbox[i][1])
                im_target_res[(i)*nt:(i+1)*nt,1] = im_target[:,1]
                im_target_res[(i)*nt:(i+1)*nt,0]=i+BOX_COUNT
            min_ = torch.min(im_target_res[...,2:3],1)[0]>=0    
            im_target_res = im_target_res[torch.min(im_target_res[...,2:10],1)[0]>=0]
            if im_target_res.shape[0]>0:
                im_target_res = im_target_res[torch.max(im_target_res[...,2:10],1)[0]<=1]
            # if im_target_res.shape[0]>0:
            #     im_target_res = im_target_res[torch.max(im_target_res[...,4:6],1)[0]<=1]
            all_batch_target.append((im_target_res))
            BOX_COUNT+=len(bbox)
            # print(im_target.shape)
    all_batch_target=torch.cat(all_batch_target,0)
    return all_batch_target


def build_targets_forlayer(p, targets):
    tcls, t_fourpoint, indices,tbox= [], [], [],[]
    all_target = get_rec_box(targets,14)
    gain = torch.ones(14, device=targets.device)
    # targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    g = 0.5
    for i in range(len(p)):
        gain[2:14] = torch.tensor(p[i].shape)[[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]]
        t = all_target * gain

        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 10:12]  # grid xy
        offsets = 0
        gxy_ = torch.round(gxy)
        gwh = t[:,12:14]

        gij = (gxy_ - offsets).long()
        gi, gj = gij.T

        pre_point_ = gxy_.repeat(1,4)

        g_fpoint = t[:, 2:10]  # grid wh

        indices.append((b, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        t_fourpoint.append(g_fpoint - pre_point_)  # box
        tcls.append(c)  # class
        tbox.append(torch.cat((gxy - gij, gwh), 1))

    return indices,t_fourpoint,tcls,tbox


def get_rec_box(target,res_shape):
    print(target.shape)
    if res_shape==14:
        res = torch.zeros((target.shape[0],res_shape))
        if target.shape[0]!=0:
            res[:,0:10] = target
            xmin = torch.min(target[:,[2,4,6,8]],1)[0]
            ymin = torch.min(target[:,[3,5,7,9]],1)[0]
            xmax = torch.max(target[:,[2,4,6,8]],1)[0]
            ymax = torch.max(target[:,[3,5,7,9]],1)[0]
            res[:,10:11] = (xmin+xmax)/2
            res[:,11:12] = (ymin+ymax)/2
            res[:,12:13] = (xmax-xmin)
            res[:,13:14] = (ymax-ymin)
        return res
    if res_shape==6:
        res = torch.zeros((target.shape[0],res_shape))
        if target.shape[0]!=0:
            res[:,0:2] = target[:,0:2]
            xmin = torch.min(target[:,[2,4,6,8]],1)[0]
            ymin = torch.min(target[:,[3,5,7,9]],1)[0]
            xmax = torch.max(target[:,[2,4,6,8]],1)[0]
            ymax = torch.max(target[:,[3,5,7,9]],1)[0]
            res[:,2:3] = ((xmin+xmax)/2).view(target.shape[0],1)
            res[:,3:4] = ((ymin+ymax)/2).view(target.shape[0],1)
            res[:,4:5] = (xmax-xmin).view(target.shape[0],1)
            res[:,5:6] = (ymax-ymin).view(target.shape[0],1)
        return res

def get_rec_box_for_predict(target):
    res = torch.zeros((target.shape[0],4))
    xmin = torch.min(target[:,[0,2,4,6]],1)[0]
    ymin = torch.min(target[:,[1,3,5,7]],1)[0]
    xmax = torch.max(target[:,[0,2,4,6]],1)[0]
    ymax = torch.max(target[:,[1,3,5,7]],1)[0]
    res[:,0] = (xmin+xmax)/2
    res[:,1] = (ymin+ymax)/2
    res[:,2] = (xmax-xmin)
    res[:,3] = (ymax-ymin)
    return res

def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    targets_box = get_rec_box(targets,6).to(device)
    tcls, tbox, indices, anchors = build_targets(p, targets_box, model)  # targets
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
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

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

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
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
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j),))
            t = t.repeat((off.shape[0], 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


