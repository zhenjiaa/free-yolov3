
import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def compute_loss(p,target,boxes):
    # p:N*85*w*h
    # target:K*6
    # boxes: N*4
    return 0