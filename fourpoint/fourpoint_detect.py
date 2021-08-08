import argparse
from enum import Flag
import time
from pathlib import Path
import sys

import numpy
from train import train
import os
sys.path.append('./')

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import nanprod, random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords_foupoint, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized




def process(img,device,img_size):
    img = letterbox(img, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = numpy.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    return img




def detect_img(model,im0s,device,imgsz,save_img):

    '''
    model ->img
    img
    '''
    img = im0s.copy()
    img = process(img,device,imgsz)
    # Inference
    t1 = time_synchronized()
    (pred,train_out),feature = model(img,refine=True)
    res,boxes = model.detector_(pred,feature)
    model.refine_net = model.refine_net.to(device)
    if boxes[0].shape[0]==0:
        return [],im0s
    res = model.refine_net(res,boxes,train=False)
    t2 = time_synchronized()
    print(t2-t1)
    res_score = res[...,8]
    index_ = torch.max(res_score,1)[1]
    res_box = []
    flag = 1
    for i in range(res.shape[0]):
        res_ = res[i,index_[i]]
        if res_[8]>0.4:
            flag=0
        else:
            continue
        res_box.append([int(res_[0]),int(res_[1]),int(res_[2]),int(res_[3]),int(res_[4]),int(res_[5]),int(res_[6]),int(res_[7])])
    if flag==1:
        return [],im0s
    

    res_box = numpy.array(res_box,dtype=numpy.float)
    shape_ = [img.shape[2],img.shape[3],3]
    print(shape_,res_box.shape,im0s.shape)
    res_box[:, :4] = scale_coords_foupoint(shape_, res_box[:, :4], im0s.shape).round()
    res_box[:, 4:8] = scale_coords_foupoint(shape_, res_box[:, 4:8], im0s.shape).round()
    if save_img:
        for res_ in res_box:
            x1 = int(res_[0])
            y1 = int(res_[1])
            x2 = int(res_[2])
            y2 = int(res_[3])
            x3 = int(res_[4])
            y3 = int(res_[5])
            x4 = int(res_[6])
            y4 = int(res_[7])

            im0s = cv2.line(im0s,(x1,y1),(x2,y2),(255,255,0),4)
            im0s = cv2.line(im0s,(x3,y3),(x4,y4),(255,255,0),4)
            im0s = cv2.line(im0s,(x3,y3),(x2,y2),(255,255,0),4)
            im0s = cv2.line(im0s,(x1,y1),(x4,y4),(255,255,0),4)

    return res_box,im0s



def detect_box(opt):
    src = opt.source
    weights = opt.weights
    imgsz = opt.img_size
    save_path = opt.save
    device = select_device(opt.device)
    
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    miss =0
    for file_ in os.listdir(src):
        file = os.path.join(src,file_)
        if file.split('.')[-1] in ['jpg','JPG','png','jpeg']:
            img_ = cv2.imread(file)
            res_box,img_ = detect_img(model,img_,device,imgsz,True)
            if res_box==[]:
                print(file_)
                miss+=1
            cv2.imwrite(os.path.join(save_path,file_),img_)
    print(miss)














if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/data1/diedai/5_0727/free-yolov3/runs/train_fourpoint_ccpd_ID1/exp/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/data1/diedai/5_0727/free-yolov3/set_train/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--save', default='yanzheng/', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_box(opt)
