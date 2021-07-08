import torch

# 不能改tensor的值再计算loss
# import torch
# x = torch.tensor(0., requires_grad=True)
# y = x.sigmoid()
# y.zero_()                   # 将y的值赋为0，
# print(y)
# y.backward()

# 可以改tensor的值再计算loss
x = torch.tensor(1., requires_grad=True)
y = x ** 2
y.detach().zero_()
print(y)
y.backward()
print(x.grad)


# 原因：两个函数求倒数不一样


#叶子节点不能执行in-place（原地）操作，因为在进行前向传播的时候得到的是叶子结点的地址，再进行反向传播的时候这个地址不变才不会报错，地址改变了就会出错
x = torch.tensor(1., requires_grad=True)
y = x ** 2
x.zero_()      
x.detach().zero_()                   
print(y)
y.backward()
print(x.grad)
