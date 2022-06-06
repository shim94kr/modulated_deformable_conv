from modulated_deform_conv import *
batch=64
cpudata=torch.ones(batch,64,5,28,28,requires_grad=True)
# data=torch.ones(batch,1,5,5,device='cuda',requires_grad=True)
data=cpudata.cuda()
offset_o=torch.ones(batch,1*5*5*5,5,28,28,device='cuda',requires_grad=True) * 1.5
offset_h=torch.ones(batch,1*5*5*5,5,28,28,device='cuda',requires_grad=True) * 1.5
offset_w=torch.ones(batch,1*5*5*5,5,28,28,device='cuda',requires_grad=True) * 1.5
offset = torch.cat([offset_o, offset_h, offset_w], dim=1)
mask=torch.ones(batch,9,5,5,device='cuda',requires_grad=True)
weight=torch.ones(64,64,5,5,5,device='cuda',requires_grad=True)
bias=torch.zeros(64,device='cuda',requires_grad=True)
stride=1
padding=2
dilation=1
groups=1
deformable_groups=1
in_step=2
'''
class DeformConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1 , in_step=64):
'''

print(data)
out=deform_conv3d(data,offset,weight,bias,stride,padding,dilation,groups,deformable_groups,in_step)
print(out)

loss=out.sum()
print(loss)
print(data.grad)
print(offset.grad)
print(weight.grad)
print(bias.grad)
loss.backward()
print(data.grad)
print(cpudata.grad)
print(bias.grad)
