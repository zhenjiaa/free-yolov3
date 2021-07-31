
from pickle import TRUE
import re
import sys
from matplotlib.pyplot import box
import numpy as np
from torch.nn.functional import upsample

from torch.nn.modules import conv
import torchvision
sys.path.append('./')
from torch.cuda import init
from models.yolo import Model
import torch.nn as nn
import torch
import torchvision.models as torch_models
import torchvision.ops as ops
# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

import cv2
import time
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv, C3
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging,non_max_suppression_refine
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class refine_head(nn.Module):
    # stride = None  # strides computed during build
    # export = False  # onnx export
    def __init__(self, nc=1,ch=512):  # detection layer
        super(refine_head, self).__init__()
        self.no = 5+nc
        self.m = nn.Conv2d(ch,self.no,1) # output conv
        
    def forward(self,x,boxes,train=True,eval=False):
        x = self.m(x)
        bs, _, ny, nx = x.shape
        x = x.permute(0,2,3,1).contiguous()   # b*n*n*5
        # if not self.train:
        # ###TODO
        if not train:
            boxes = boxes[0]
            y = x.sigmoid()
            self.grid = self._make_grid(nx, ny).to(x.device)
            y[..., 0:2] = ((y[...,0:2] * 2. - 0.5 )+ self.grid).to(x.device)
            
            y[..., 0] = y[...,0]*(boxes[:,2]-boxes[:,0]).view(-1,1,1)/y.shape[1]+boxes[:,0].view(-1,1,1)
            y[..., 1] = y[...,1]*(boxes[:,3]-boxes[:,1]).view(-1,1,1)/y.shape[2]+boxes[:,1].view(-1,1,1)
            y[..., 2] = (y[..., 2])*(boxes[:,2]-boxes[:,0]).view(-1,1,1)
            y[..., 3] = (y[..., 3])*(boxes[:,3]-boxes[:,1]).view(-1,1,1)
            y = y.contiguous().view(bs,-1,y.shape[-1])
            # y[..., 4] = torch.ones_like(y[..., 4]).to(x.device)
            return y
        if eval:
            boxes_ = torch.cat(boxes,0)
            # num_b = boxes_.shape[1]
            y = x.sigmoid()
            self.grid = self._make_grid(nx, ny).to(x.device)
            y[..., 0:2] = ((y[...,0:2] * 2. - 0.5 )+ self.grid).to(x.device)
            
            y[..., 0] = y[...,0]*(boxes_[:,2]-boxes_[:,0]).view(-1,1,1)/y.shape[1]+boxes_[:,0].view(-1,1,1)
            y[..., 1] = y[...,1]*(boxes_[:,3]-boxes_[:,1]).view(-1,1,1)/y.shape[2]+boxes_[:,1].view(-1,1,1)
            y[..., 2] = (y[..., 2])*(boxes_[:,2]-boxes_[:,0]).view(-1,1,1)
            y[..., 3] = (y[..., 3])*(boxes_[:,3]-boxes_[:,1]).view(-1,1,1)
            y = y.contiguous().view(bs,-1,y.shape[-1])
            return y
            # y[..., 4] = torch.ones_like(y[..., 4]).to(x.device)


        return [x] if self.train else y


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()


class refine_net(nn.Module):
    def __init__(self,cfg,ch=32):
        super(refine_net, self).__init__()
        print(cfg)
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            with open(cfg) as f:
                import yaml
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.ch = ch
        nc = self.yaml['nc']
        self.rf_model,last_ch = self.parse_model()
        self.rf_model = self.rf_model
        self.rf_head = refine_head(nc=nc,ch=torch.tensor(self.forward_compute_s(ch)))
    
    def forward_compute_s(self,ch):
        z = 128
        x = torch.zeros(1,ch,z,z)
        for i,layer in enumerate(self.rf_model):
            x = layer(x)
        print('downsample',z/x.shape[-1])
        return x.shape[1]
    
    def forward(self,x,boxes=[],train=True,eval=False):
        # print(self.rf_model)
        # exit()
        for i,layer in enumerate(self.rf_model):
            x = layer(x)
        x = self.rf_head(x,boxes,train,eval)

        return x
    
    def parse_model(self):
        modellist= []
        ch = self.ch
        res_mod = torchvision.models.resnet18()
        k_ = [res_mod.layer1,res_mod.layer2,res_mod.layer3]
        for i,(f,n,m,args) in enumerate(self.yaml['refine_net']):
            if m =='Conv':
                per_conv = Conv(ch,args[0],args[1],args[2])
                ch = args[0]
            elif m=='nn.Upsample':
                per_conv = nn.Upsample(None,2,'nearest')
            elif m =='resblock':
                per_conv = k_[args[0]]
            elif m =='maxpool':
                per_conv = res_mod.maxpool
            modellist.append(per_conv)
        return nn.Sequential(*modellist),ch


class detector():
    def __init__(self,detector_args) -> None:
        self.conf_thres = detector_args['conf_thres']
        self.iou_thres = detector_args['iou_thres']
    
    def __call__(self,pred__,feature,train_=False):
        # detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        # bs = len(pr)
        # ig = (feature[0][0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.int)
        # im_name  = str(time.time())+'.jpg'
        # cv2.imwrite('yanzheng/'+im_name,ig)
        feature=feature[0]
        preds = non_max_suppression_refine(pred__, self.conf_thres, self.iou_thres, classes=None)
        # print(self.conf_thres, self.iou_thres)
        pic_w,pic_h = feature.shape[2],feature.shape[3]
        
        
        boxes = []
        for i,pred_ in enumerate(preds):
            # print(torch.max(pred_[:,4]),pred_[0,4:])
            pred = pred_[:,0:4]
            if pred.shape[0]>25:
                pred=pred[0:25,:]
            # pred = torch.tensor([[0,160,160,320]]).to(feature.device).float()
            prd = torch.zeros_like(pred).to(feature.device)
            if pred.shape[0]!=0:
                w = (pred[:,2]-pred[:,0]).unsqueeze(0)
                h = (pred[:,3]-pred[:,1]).unsqueeze(0)
                c_y = (pred[:,3]+pred[:,1])/2
                c_x = (pred[:,2]+pred[:,0])/2
                # if train_:
                    # c_y = c_y+w*
                wh = torch.cat((w,h),0)
                wh = torch.max(wh,0)[0]*2
                prd[:,3] = (c_y+wh/2).clamp(0,pic_h-1)
                prd[:,1] = (c_y-wh/2).clamp(0,pic_h-1)
                prd[:,2] = (c_x+wh/2).clamp(0,pic_w-1)
                prd[:,0] = (c_x-wh/2).clamp(0,pic_w-1)

                # pred[]
                # pred = torch.tensor([[20.,20.,100.,100.]]).to(device)
                # print(pred.shape)
            boxes.append(prd)

        per_fear = ops.roi_align(feature,boxes,[32,32])
        # ig = (per_fear[0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.int)
        # im_name  = +'.jpg'
        # cv2.imwrite('yanzheng/'+im_name,ig)
        return per_fear,boxes


class refine_yolo(Model):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None ,detector_args=None): 
        Model.__init__(self, cfg, ch, nc=nc)
        # self.get_
        self.refine_net = refine_net(cfg,ch=32)
        self.detector_ = detector(detector_args)

   
        # self.detector = detector(detector_args)



if __name__ == '__main__':
    import sys
    sys.path.append('../')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    # opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    detector_args={}
    detector_args['conf_thres']=0.00001
    detector_args['iou_thres']=0.6
    model = refine_yolo('cfg/ccpd/refine_res16.yaml',detector_args=detector_args).to(device)
    model.train()
    # print(model)
    x = torch.rand((8,3,320,320)).to(device)
    # x = cv2.imread('/data1/paper_/test/free-yolov3/coco128/images/train2017/000000000009.jpg')
    # x = torch.tensor(cv2.resize(x,(320,320))).to(device)
    # x = x.permute(2,0,1).unsqueeze(0).float()
    (detect_res,pred),feature = model(x,refine=True)
    # res,boxes = model.detector_(detect_res,[x])
    # print(model)
    # if model.training:
    #     res,_ = model.detector_(detect_res,feature)
    #     res = model.refine_net(res)
    # else:
    # exit()
    res,boxes = model.detector_(detect_res,feature)
    model.refine_net=model.refine_net.to(device)
    res = model.refine_net(res,boxes)
    print(res[0].shape)
    # print(boxes.shape)









        
        


