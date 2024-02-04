import csv
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import KFold
import math
from sympy import *
import matplotlib.pyplot as plt


fn1='./p1_right/Ds_pca5.csv'
fn2='./p1_right/Dt_pca5.csv'
fn3='./p1_right/kmeans_mxpt_src5.csv'
fn4='./p1_right/kmeans_mxpt_tar5.csv'
fn5='./p1_right/A_nolabel/A3_4.csv'  #A saved last time
fn6='./p1_right/A_nolabel/A3_5.csv'

pretrained=1
pca=1
lamb1=0.5
lamb2=0.5
lr=4e-2
epochs=8000
hand='right'

if hand=='right':
    len_data=17
    L=['y','u','h','j','n','m','i','k','l','o','p','+']
else:
    len_data=14

if pca==1:
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

if pca==1:
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


#de-mean Ds, Dt
means=[]
meant=[]
stds=[]
stdt=[]
Ds_or=Ds  #before centered
for j in range(len(Ds[0])):
    means.append(np.mean(Ds[:,j]))
    meant.append(np.mean(Dt[:,j]))

    for i in range(len(Ds)):
        Ds[i,j]=(Ds[i,j]-means[j])
    for i in range(len(Dt)):
        Dt[i,j]=(Dt[i,j]-meant[j]) 
    for i in range(len(Ks)):
        Ks[i,j]=(Ks[i,j]-means[j])
    for i in range(len(Kt)):
        Kt[i,j]=(Kt[i,j]-meant[j]) 


Ds=np.array(Ds,dtype=float)
print('Dt.shape:',Dt.shape)
print('Ds.shape:',Ds.shape)
print('np.size(Ds,0):',np.size(Ds,0))  #1 means column_num, 0 means row_num

Cs=np.cov(Ds,rowvar=False)
Ct=np.cov(Dt,rowvar=False)

A1=np.dot(fractional_matrix_power(Cs, -0.5),fractional_matrix_power(Ct, 0.5))
A2=np.dot(fractional_matrix_power(np.dot(Ks.T,Ks),-1),np.dot(Ks.T,Kt))
##

if pretrained==1:
    with open(fn5,"r") as f5:
        A0 = list(csv.reader(f5))
    A0=np.array(A0,dtype=float)
else:
    A0=A2
st=0

#compare f2-norm of A0 and Ks
def f2norm(B):
    sum=0
    for i in range(len(B)):
        for j in range(len(B[0])):
            sum+=B[i][j]*B[i][j]
    return sum
sumks=f2norm(Ks)
print('validate the success of loss, f2(Ks):',sumks)


def loss(A):
    a=np.dot(np.dot(A.T,Cs),A)-Ct  #coral loss before norm
    b=np.dot(Ks,A)-Kt  
    lossw=0
    loss1=0
    loss2=0
    # st=0
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
    d2=2*lamb2*np.dot(Ks.T,np.dot(Ks,A))-2*lamb2*np.dot(Ks.T,Kt)  
    #
    d1=4*np.dot(np.dot(Cs,np.dot(A,A.T)),np.dot(Cs,A))-4*np.dot(Cs,np.dot(A,Ct))
    d1=d1*lamb1
    
    return d1+d2

globalA=[]
loss1w=[]
loss2w=[]
lossw=[]
globalA.append(A0)
A=A0
lr1=lr
print('loss1 orgin:',loss(A))
for k in range(epochs):
    temA = A - lr * dA(A)
    A = temA
    globalA.append(temA)
    whole_l,ls1,ls2=loss(A)
    print(k,'  loss:',whole_l)
    if whole_l>10e12:
        print('loss out of range')
        break

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

with open(fn6,'a+',newline="") as f6:
    writer=csv.writer(f6)
    for line in A:
            writer.writerow(line)
