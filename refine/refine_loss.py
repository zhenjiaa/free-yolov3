
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
    b_boxes = torch.cat(boxes,0)
    indices,tbox,tcls = build_targets_forlayer(p, targets)
    
    bs = p[0][...,0].shape[0]
    
    # # print(bs)
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
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
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # print(pi.shape)
                pwh = (ps[:, 2:4].sigmoid()*2).to(device)
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, DIoU=True,CIoU=True)  # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()

            # Objectness
            tobj[b, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

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
    loss = lbox +lobj+lcls
        # loss = loss*3
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()



    
    # p:N*85*w*h
    # target:K*6
    # boxes: N*4
    return 0


def build_targets_forbatch(feature_size,target,bboxes):
    batch_size = len(bboxes)
    all_batch_target = []
    for j in range(batch_size):
        BOX_COUNT = 0
        im_target = target[target[:,0]==j]
        nt = im_target.shape[0]
        if nt:
            bbox = bboxes[j]
            im_target[:,2] = im_target[:,2]*feature_size[0]
            im_target[:,3] = im_target[:,3]*feature_size[1]
            im_target[:,4] = im_target[:,4]*feature_size[0]
            im_target[:,5] = im_target[:,5]*feature_size[1]
            im_target = im_target.repeat(len(bbox),1)
            for i in range(len(bbox)):
                im_target[(i)*nt:(i+1)*nt,2] = (im_target[(i)*nt:(i+1)*nt,2]-bbox[i][0])/(bbox[i][3]-bbox[i][1])
                im_target[(i)*nt:(i+1)*nt,3] = (im_target[(i)*nt:(i+1)*nt,3]-bbox[i][1])/(bbox[i][3]-bbox[i][1])
                im_target[(i)*nt:(i+1)*nt,4] = im_target[(i)*nt:(i+1)*nt,4]/(bbox[i][2]-bbox[i][0])
                im_target[(i)*nt:(i+1)*nt,5] = im_target[(i)*nt:(i+1)*nt,5]/(bbox[i][3]-bbox[i][1])
                im_target[(i)*nt:(i+1)*nt,0]=i+BOX_COUNT
            im_target = im_target[torch.min(im_target[...,2:3],1)[0]>=0]
            im_target = im_target[torch.max(im_target[...,2:3],1)[0]<=1]
            all_batch_target.append((im_target))
            BOX_COUNT+=len(bbox)
            # print(im_target.shape)
    all_batch_target=torch.cat(all_batch_target,0)
    return all_batch_target


def build_targets_forlayer(p, targets):
    tcls, tbox, indices = [], [], []
    gain = torch.ones(6, device=targets.device)
    # targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    g = 0.5
    off = torch.tensor([[0, 0],
                    ], device=targets.device).float() * g
    for i in range(len(p)):
        gain[2:6] = torch.tensor(p[i].shape)[[2, 1, 2, 1]]
        t = targets * gain
        gxy = t[:, 2:4]  # grid xy
        gxi = gain[[2, 3]] - gxy  # inverse

        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        offsets = 0
        gxy_ = torch.round(gxy)

        gwh = t[:, 4:6]  # grid wh

        gij = (gxy_ - offsets).long()
        offsets = 0  
        gi, gj = gij.T  # grid xy indices

        indices.append((b, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        tcls.append(c)  # class
    return indices,tbox,tcls