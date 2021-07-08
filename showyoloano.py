import os
import cv2


def showpic(imgfile,lines,file):
    print(imgfile)
    img=cv2.imread(imgfile)
    width=img.shape[1]
    height=img.shape[0]
    # print(width,height)
    for line in lines:
        line=line.split('\n')[0].split(' ')
        # print(line[1])
        xmin=int((float(line[1])-float(line[3])/2)*width)
        xmax=int((float(line[1])+float(line[3])/2)*width)
        ymin=int((float(line[2])-float(line[4])/2)*height)
        ymax=int((float(line[2])+float(line[4])/2)*height)
        if xmin<0 or ymin<0:
            print(line)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),5) # green
    # cv2.namedWindow('cvcv', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('cvcv', 1024,768);
    # cv2.imshow('cvcv',img)
    # cv2.waitKey(1000)
    cv2.imwrite('/data1/paper_/test/free-yolov3/val/res/'+file,img)
    # cv2.destroyAllWindows()
    # exit()





path='/data1/paper_/test/free-yolov3/val/'
for root,dir,files in os.walk(path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('png'):
            txtfile=root+'/'+file[0:-4]+'.txt'
            imgfile=root+"/"+file

            with open(txtfile,'r') as f:
                f=f.readlines()
                showpic(imgfile,f,file)