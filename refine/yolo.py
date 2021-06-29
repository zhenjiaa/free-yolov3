
import sys
sys.path.append('./')
from torch._C import set_flush_denormal
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
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class refine_net(nn.Module):
    def __init__(self):
        super(refine_net, self).__init__()
        self.backbone = torch_models.resnet18()
    def forward(x):
        return self.backbone(x)

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
        preds = torch.cat((preds),0)
        print(feature.shape)
        pic_w,pic_h = feature.shape[2],feature.shape[3]
        
        
        boxes = []
        for i,pred in enumerate(preds):
            # pred = pred.int()
            pred = pred[:,0:4]
            w = pred[:,2]-pred[:,0]
            h = pred[:,3]-pred[:,1]
            c_y = (pred[:,3]+pred[:,1])/2
            c_x = (pred[:,2]+pred[:,0])/2
            pred[:,1] = (c_y-w/2).clamp(0,pic_h).int()
            pred[:,3] = (c_y+w/2).clamp(0,pic_h).int()
            pred[:,0] = pred[:,0].clamp(0,pic_w).int()
            pred[:,2] = pred[:,2].clamp(0,pic_w).int()
        
        # pred = pred[...,0:4]
        # w = pred[...,2]-pred[...,0]
        # h = pred[...,3]-pred[...,1]
        # c_y = (pred[...,3]+pred[...,1])/2
        # c_x = (pred[...,2]+pred[...,0])/2
        # pred[...,1] = (c_y-w/2).clamp(0,pic_h).int()
        # pred[...,3] = (c_y+w/2).clamp(0,pic_h).int()
        # pred[...,0] = pred[...,0].clamp(0,pic_w).int()
        # pred[...,2] = pred[...,2].clamp(0,pic_w).int()
        


            
        pass

        per_fear = ops.roi_align(feature[i],pred,[64,64])
            
        return pred
        pass


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
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    x = torch.rand((8,3,320,320)).to(device)
    (detect_res,pred),feature = model(x,refine=True)
    res = model.detector_(detect_res,feature)









        
        


