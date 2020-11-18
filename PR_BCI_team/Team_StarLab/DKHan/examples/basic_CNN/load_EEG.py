import scipy.io as sio
import numpy as np
import os

try:
    del smt_train # does a exist in the current namespace
    del y
except NameError:
    smt_train = None
    y = None

sess = 1
#for문 끝숫자는 포함 안된다 ㅋㅋ 전체 사용자의경우 55로 해야함
for i in range(1,2):
    dataname = 'sess%02d_subj%02d_EEG_MI.mat' % (sess, i)
    path = 'C:/Data_MI/' + dataname
    MI_s1 = sio.loadmat(path,struct_as_record=False,squeeze_me=True)

    temp = MI_s1['EEG_MI_train']
    temp2 = temp.x

    #temp2 = torch.tensor(temp.smt).unsqueeze(3)
    #temp2 = torch.tensor(temp.smt)
    #if i > 0:
    #    smt_train = torch.cat([smt_train, temp2], dim=1)
    #else:
    #    smt_train = temp2
    #smt_train.shape

    temp = MI_s1['EEG_MI_train']
    temp2 = torch.tensor(temp.y_logic)
    if i > 0:
        y = torch.cat([y, temp2], dim=1)
    else:
        y = temp2

x = smt_train.permute(0,2,1) #trial 뒤로 뺌 ex torch.Size([4000, 62, 200]) 2명 




   #데이터를 10명쯤 로드하면 한 6기가 먹는거 같은데
   #적어도 트라이얼마다는 쪼개서 저장해서 배치로더를 만들어야 할거 같다.
   #꼼지락꼼지락...
'''
ddd = torch.randn(1)
if torch.cuda.is_available():
    device = torch.device("cuda")
    dd = torch.ones_like(ddd,device=device)
    ddd = ddd.to(device)
    z = dd+ddd
    print(z)
    print(z.to("cpu",torch.double))

dd = torch.ones(2,2,requires_grad = True)
print(dd)
ddd = dd+2
print(ddd)
print(ddd.grad_fn)

z = ddd*ddd*3

out = z.mean()
out.backward()
print(out)
print(dd.grad)
'''

