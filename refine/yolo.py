
import sys
import numpy as np
from torch.nn.functional import upsample

from torch.nn.modules import conv
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
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv, C3
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging,non_max_suppression
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class refine_head(nn.Module):
    # stride = None  # strides computed during build
    # export = False  # onnx export
    def __init__(self, nc=1,ch=512,up =1):  # detection layer
        super(refine_head, self).__init__()
        self.no = 5+nc
        self.m = nn.Conv2d(ch,self.no,1) # output conv
        self.upsample = up
        
    def forward(self,x,boxes):
        x = self.m(x)
        x = x.permute(0,2,3,1)   # b*n*n*5
        # if not self.train:

        # evalu ?
        if not self.train:
            y = x.sigmoid()

        ###TODO
        # y[..., 0] = (y[..., 0:0] * 2. - 0.5 )*(boxes[:,2]-boxes[:,0])/64   # x
        # y[..., 1] = (y[..., 0:1] * 2. - 0.5 )*(boxes[:,3]-boxes[:,1])/64   # y

        return [x]


class refine_net(nn.Module):
    def __init__(self,cfg='models/yolov3.yaml',ch=32):
        super(refine_net, self).__init__()
        with open(cfg) as f:
            import yaml
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.ch = ch
        nc = self.yaml['nc']
        self.rf_model,last_ch = self.parse_model()
        self.rf_model = self.rf_model
        self.rf_head = refine_head(nc=nc,ch=last_ch,up=torch.tensor(self.forward_compute_s(ch)))
    
    def forward_compute_s(self,ch):
        z = 128
        x = torch.zeros(1,ch,z,z)
        for i,layer in enumerate(self.rf_model):
            x = layer(x)
        return z/x.shape[-1]
    
    def forward(self,x,boxes=[]):
        # print(self.rf_model)
        # exit()
        for i,layer in enumerate(self.rf_model):
            print('xxx')
            x = layer(x)
        x = self.rf_head(x,boxes)
        return x
    
    def parse_model(self):
        modellist= []
        ch = self.ch
        for i,(f,n,m,args) in enumerate(self.yaml['refine_net']):
            if m =='Conv':
                per_conv = Conv(ch,args[0],args[1],args[2])
                ch = args[0]
            modellist.append(per_conv)
        return nn.Sequential(*modellist),ch


class detector():
    def __init__(self,detector_args) -> None:
        self.conf_thres = detector_args['conf_thres']
        self.iou_thres = detector_args['iou_thres']
        self.classes = detector_args['classes']
    
    def __call__(self,pred,feature):
        # detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        # bs = len(pr)
        feature=feature[0]
        preds = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)
        pic_w,pic_h = feature.shape[2],feature.shape[3]
        
        
        boxes = []
        for i,pred_ in enumerate(preds):
            pred = pred_[:,0:4]
            if pred.shape[0]!=0:
                print(torch.max(pred_[:,4],0))
                if pred.shape[0]>100:
                    pred=pred[0:100,:]
                w = (pred[:,2]-pred[:,0]).unsqueeze(0)
                h = (pred[:,3]-pred[:,1]).unsqueeze(0)
                c_y = (pred[:,3]+pred[:,1])/2
                c_x = (pred[:,2]+pred[:,0])/2
                wh = torch.cat((w,h),0)
                wh = torch.max(wh,0)[0]*2
                pred[:,3] = (c_y+wh/2).clamp(0,pic_h-1)
                pred[:,1] = (c_y-wh/2).clamp(0,pic_h-1)
                pred[:,2] = (c_x+wh/2).clamp(0,pic_w-1)
                pred[:,0] = (c_x-wh/2).clamp(0,pic_w-1)
                # pred[]
                # pred = torch.tensor([[20.,20.,100.,100.]]).to(device)
                # print(pred.shape)
            boxes.append(pred.to(device))
        
        per_fear = ops.roi_align(feature,boxes,[64,64])
        # cv2.imwrite('yanzheng/2.jpg',ig)
        return per_fear,boxes


class refine_yolo(Model):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None ,detector_args=None): 
        Model.__init__(self, cfg, ch=3, nc=nc)
        # self.get_
        self.refine_net = refine_net()
        self.detector_ = detector(detector_args)

   
        # self.detector = detector(detector_args)



if __name__ == '__main__':
    import sys
    sys.path.append('../')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    # opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    detector_args={}
    detector_args['conf_thres']=0.00001
    detector_args['iou_thres']=0.6
    detector_args['classes']=1
    model = refine_yolo('models/yolov3.yaml',detector_args=detector_args).to(device)
    model.train()
    # print(model)
    x = torch.rand((8,3,32,32)).to(device)
    (detect_res,pred),feature = model(x,refine=True)
    res,boxes = model.detector_(detect_res,feature)
    # if model.training:
    #     res,_ = model.detector_(detect_res,feature)
    #     res = model.refine_net(res)
    # else:
    res,boxes = model.detector_(detect_res,feature)
    model.refine_net=model.refine_net.to(device)
    res = model.refine_net(res,boxes)
    print(res.shape)
    print(boxes.shape)









        
        


