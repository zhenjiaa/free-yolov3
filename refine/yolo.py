from torch._C import set_flush_denormal
from torch.cuda import init
from models.yolo import Model
import torch.nn as nn
import torch
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
        self.module= nn.ModuleList([
            nn.Linear(1,2),
            nn.Linear(1,2),
            nn.Linear(1,2)
        ])
    def forward(x):
        return x
        pass

class detector():
    def __init__(self,detector_args) -> None:
        self.conf_thres = detector_args['conf_thres']
        self.iou_thres = detector_args['iou_thres']
        self.classes = detector_args['classes']
    
    def __call__(self,pred):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)
        return pred
        pass


class refine_yolo(Model):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None ,detector_args=None): 
        Model.__init__(self, cfg, ch=3, nc=nc)
        # self.get_
        self.refine_net = refine_net()
        self.detector_ = detector(detector_args)
        # self.detector = detector(detector_args)






        
        


