import torch

# 不能改tensor的值再计算loss
v = torch.meshgrid([torch.arange(10)])
print(v)
print(torch.stack((v), 1).view((1, 1, 10, 1)).float())



import shutil
for i in open('part_val.txt','r'):
    print(i)
    i = i.strip()
    name = i.split('/')[-1]
    shutil.copy(i,'val/'+name)
