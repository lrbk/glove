import csv
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import KFold
import math
from sympy import *
import matplotlib.pyplot as plt

# fn1='./hanxue_right0811/Ds_pca14.csv'  #Ds
# fn2='./hanxue_right0811/Dt_pca14.csv'  #Dt
# fn3='./p6_data/after_pre/kmeans_mxpt_r2.csv'
# fn4='./hanxue_right0811/kmeans_mxpt_r2.csv'

# fn5='./hanxue_right0811/A2_12_split895/A5_6.csv'  #上一轮跑时储存的A  A10A11已经跑过了
# fn6='./hanxue_right0811/A2_12_split895/A5_7.csv'
fn1='./p15_right0109/Ds_pca5.csv'
fn2='./p15_right0109/Dt_pca5.csv'
fn3='./p15_right0109/kmeans_mxpt_src5.csv'
fn4='./p15_right0109/kmeans_mxpt_tar5.csv'
fn5='./p15_right0109/A_nolabel/A3_4.csv'
fn6='./p15_right0109/A_nolabel/A3_5.csv'
# fn1='./hanxue_left0825/Ds_pca1.csv'  #Ds
# fn2='./hanxue_left0825/Dt_pca1.csv'  #Dt
# fn3='./hanxue_left0825/kmeans_mxpt_r1.csv'
# # fn4='./hanxue_right0530/kmeans_d6.csv'
# fn4='./p6_data/after_pre/kmeans_mxpt_r1.csv'

# fn5='./hanxue_left0825/A/A1_8.csv'  #上一轮跑时储存的A  A10A11已经跑过了
# fn6='./hanxue_left0825/A/A1_9.csv'
pretrained=1
pca=1
lamb1=0.5
lamb2=0.5
lr=4e-2
epochs=16000
hand='right'


# print('test:')
# test=np.array([[2,0],[0,0]])
# print(test)
# print('-1:',fractional_matrix_power(test,-1))

test_rotate=0
if test_rotate==1:
    fn1='./hanxue_right0811/plot_d6.csv'
    # fn2='./hanxue_right0811/plot_l6.csv'
    fn2='./hanxue_right0811/test_d6.csv'       

    fn3='./hanxue_right0811/kmeans_d2.csv'
    fn4='./p6_data/after_pre/kmeans_d2.csv'


nsplits=1


if hand=='right':
    len_data=17
    L=['y','u','h','j','n','m','i','k','l','o','p','+']
else:
    len_data=14

if test_rotate==1 or pca==1:
    with open(fn1,"r") as f1:
        Ds0 = list(csv.reader(f1))
    Ds=np.array(Ds0,dtype=float)
else:
    Ds0 = pd.read_csv(fn1,header=None)
    Ds=[]
    Ds2=[]
    num=int(len(Ds0)/len_data)
    print('num:',num)
    for i in range(0,num):
        x=[]
        for j in range(0,8):#We have 8 sensors
            height=Ds0.iloc[len_data*i:len_data*i+len_data,j].to_list()
            x.extend(height)
        Ds.append(x)
    Ds=np.array(Ds,dtype=float)

if test_rotate==1 or pca==1:
    with open(fn2,"r") as f2:
        Dt = list(csv.reader(f2))
    Dt=np.array(Dt,dtype=float)
else:
    Dt0 = pd.read_csv(fn2,header=None)
    Dt=[]
    num=int(len(Dt0)/len_data)
    print('num:',num)
    for i in range(0,num):
        x=[]
        for j in range(0,8):#We have 8 sensors
            height=Dt0.iloc[len_data*i:len_data*i+len_data,j].to_list()
            x.extend(height)
        Dt.append(x)
    Dt=np.array(Dt,dtype=float)

with open(fn3,"r") as f3:
    Ks = list(csv.reader(f3))
Ks=np.array(Ks,dtype=float)
with open(fn4,"r") as f4:
    Kt = list(csv.reader(f4))
Kt=np.array(Kt,dtype=float)


#Ds，Dt去均值
means=[]
meant=[]
stds=[]
stdt=[]
Ds_or=Ds  #去中心化前
for j in range(len(Ds[0])):
    means.append(np.mean(Ds[:,j]))
    meant.append(np.mean(Dt[:,j]))
    # stds.append(np.std(Ds[:,j]))
    # stdt.append(np.std(Ds[:,j]))
    for i in range(len(Ds)):
        # Ds[i,j]=(Ds[i,j]-means[j])/stds[j]
        Ds[i,j]=(Ds[i,j]-means[j])
    for i in range(len(Dt)):
        Dt[i,j]=(Dt[i,j]-meant[j])  #/stdt[j]
    for i in range(len(Ks)):
        Ks[i,j]=(Ks[i,j]-means[j])  #/stds[j]
    for i in range(len(Kt)):
        Kt[i,j]=(Kt[i,j]-meant[j])  #/stdt[j]
     

# with open(fn4,"r") as f4:
#     y0 = list(csv.reader(f4))

# kf = KFold(n_splits=nsplits)
# for train,test in kf.split(Ds2):
#     Ds=[]
#     Ds_y=[]
#     for val in test:
#         Ds.append(Ds2[val])
#         Ds_y.append(y0[val]) 
#     break
# # print('Ds_y:',Ds_y)
# print('len(Ds_y):',len(Ds_y))
# with open(fn5,'a+',newline="") as f5:
#     writer=csv.writer(f5)
#     for line in Ds_y:
#             writer.writerow(line)


Ds=np.array(Ds,dtype=float)
# Dt=np.mat(Dt)
# print('Dt:',Dt)
# print('Ds:',Ds)
print('Dt.shape:',Dt.shape)
print('Ds.shape:',Ds.shape)
print('np.size(Ds,0):',np.size(Ds,0))  #1 means column_num, 0 means row_num


Cs=np.cov(Ds,rowvar=False)
# print('Cs11:',Cs11)

Ct=np.cov(Dt,rowvar=False)
# Cs_1=np.linalg.det(Cs)
# print('Cs_1:',Cs_1)
# print('A:',A)
# if A_1 !=0:


###########################
# d=np.size(Ds,1)
# ns=np.size(Ds,0)
# nt=np.size(Dt,0)
# tmp_s = np.dot(np.eye(1, ns),Ds)
# Cs = (Ds.T @ Ds - (tmp_s.T @ tmp_s) / ns) / (ns - 1)
# # print('Cs:',Cs)
# # print(Cs11.shape)
# # print(Cs.shape)

# # target covariance
# tmp_t = np.eye(1, nt) @ Dt
# Ct = (Dt.T @ Dt - (tmp_t.T @ tmp_t) / nt) / (nt - 1)
################################

# Ds=np.mat(np.dot(Ds,fractional_matrix_power(Cs, -0.5)))
# Ds1=(np.dot(Ds,fractional_matrix_power(Ct, 0.5))).A
# print('Ds1.shape:',Ds1.shape)
# print(Ds1[0])

A1=np.dot(fractional_matrix_power(Cs, -0.5),fractional_matrix_power(Ct, 0.5))
# print('A1:',A1)

# print('Cs-Cs.T:',Cs-Cs.T)
# print('Ct-Ct.T:',Ct-Ct.T)

###################################################################
A2=np.dot(fractional_matrix_power(np.dot(Ks.T,Ks),-1),np.dot(Ks.T,Kt))

##

if pretrained==1:
    with open(fn5,"r") as f5:
        A0 = list(csv.reader(f5))
    A0=np.array(A0,dtype=float)
else:
    A0=A2
st=0

#比较A0和Ks的f-norm
def f2norm(B):
    sum=0
    for i in range(len(B)):
        for j in range(len(B[0])):
            sum+=B[i][j]*B[i][j]
    return sum
sumks=f2norm(Ks)
print('验证loss的成功性，f2(Ks):',sumks)


def loss(A):
    a=np.dot(np.dot(A.T,Cs),A)-Ct  #coral loss before norm
    b=np.dot(Ks,A)-Kt  
    lossw=0
    loss1=0
    loss2=0
    st=0
    for i in range(len(a)):
        for j in range(len(a[0])):
            loss1+=a[i][j]*a[i][j]
    loss1=loss1*lamb1
    print('loss1:',loss1)
    for i in range(len(b)):
        for j in range(len(b[0])):
            loss2+=b[i][j]*b[i][j]
    loss2=loss2*lamb2
    print('loss2:',loss2)
    lossw=loss1+loss2
    # if loss1>0.05:
    #     st=1
    return lossw,loss1,loss2

def dA(A):
    # d=4*np.dot(np.dot(Cs,np.dot(A,A.T)),np.dot(Cs,A))-4*np.dot(Cs,np.dot(A,Ct))+2*lamb*np.dot(Ks.T,np.dot(Ks,A))-2*lamb*np.dot(Ks.T,Kt)
    d2=2*lamb2*np.dot(Ks.T,np.dot(Ks,A))-2*lamb2*np.dot(Ks.T,Kt)  
    #
    d1=4*np.dot(np.dot(Cs,np.dot(A,A.T)),np.dot(Cs,A))-4*np.dot(Cs,np.dot(A,Ct))
    d1=d1*lamb1
    #

    # meand=[]
    # stdd=[]
    # for j in range(len(d1[0])):
    #     meand.append(np.mean(d1[:,j]))
    #     stdd.append(np.std(d1[:,j]))
    #     for i in range(len(d1)):
    #         d1[i,j]=(d1[i,j]-meand[j])/stdd[j]
    # print('d:',d)
    # d=d/200
    return d1+d2

globalA=[]
loss1w=[]
loss2w=[]
lossw=[]
globalA.append(A0)
# A=np.eye(136, dtype=float)
A=A0
lr1=lr
print('loss1 orgin:',loss(A))
for k in range(epochs):
    # if k>150:
    #     lr=lr1/10

    temA = A - lr * dA(A)
    A = temA
    # print(k,'  A:',A)
    globalA.append(temA)
    whole_l,ls1,ls2=loss(A)
    print(k,'  loss:',whole_l)
    # if st1==1:
    #     break
    # if whole_l<0.1:
    #     break
    if whole_l>10e12:
        print('loss out of range')
        break
    # if whole_l<0.01*sumks:
    #     print('loss/f2norm(Ks)<0.01:')

    loss1w.append(ls1)
    loss2w.append(ls2)
    lossw.append(whole_l)

loss1w=np.array(loss1w)
loss2w=np.array(loss2w)
lossw=np.array(lossw)
xx=np.linspace(0,k,k+1)
plt.plot(xx,loss1w,color='blue')
plt.plot(xx,loss2w,color='green')
plt.plot(xx,lossw,color='red')
plt.legend(['loss1','loss2','whole loss'])
# plt.title()
plt.show()

# A=[[-0.25429986,-0.96712542],[ 0.96712542,-0.25429986]]

# Ds1=np.dot(Ds,A)

# if pretrained==1:
with open(fn6,'a+',newline="") as f6:
    writer=csv.writer(f6)
    for line in A:
            writer.writerow(line)

# ####################################
# x=Ds


# label_path='./hanxue_right0811/plot_l6.csv'
# with open(label_path,"r") as f7:
#     y = list(csv.reader(f7))
# y = [i for ii in y for i in ii]
# y=np.array(y,dtype=object)


# centers = [[2,2],[8,2],[2,8]]
# colors = ['red', 'green','yellow']
# labels=['0','1','2']

# plt.subplot(131)
# for i in range(3):  # shape[] 类别的种类数量(2)
#     plt.scatter(x[y == str(i), 0],  # 横坐标
#                 x[y == str(i), 1],  # 纵坐标
#                 c=colors[i],  # 颜色
#                 label=labels[i])  # 标签

# plt.legend()  # 显示图例

# plt.subplot(132)
# for i in range(3):  # shape[] 类别的种类数量(2)
#     plt.scatter(Ds1[y == str(i), 0],  # 横坐标
#                 Ds1[y == str(i), 1],  # 纵坐标
#                 c=colors[i],  # 颜色
#                 label=labels[i])  # 标签
# plt.legend()  # 显示图例

# plt.subplot(133)
# for i in range(3):  # shape[] 类别的种类数量(2)
#     plt.scatter(Dt[y == str(i), 0],  # 横坐标
#                 Dt[y == str(i), 1],  # 纵坐标
#                 c=colors[i],  # 颜色
#                 label=labels[i])  # 标签
# plt.legend()  # 显示图例


# plt.show() 





# # with open(fn5,'a+',newline="") as f5:
# #     writer=csv.writer(f5)
# #     for line in Ds1:
# #             writer.writerow(line)



# # print('A.T*Cs*A:',np.dot(np.dot(A_coral.T,Cs),A_coral))
# # print('Ct:',Ct)
# # X=np.dot(np.dot(A_coral.T,Cs),A_coral)-Ct
# # print('f norm:',np.linalg.norm(X))