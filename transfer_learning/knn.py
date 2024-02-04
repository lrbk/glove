#除了kmeans取中心点，还有最大点,只用2个标签
from sklearn.cluster import KMeans
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import os

fn1='./hanxue_right0811/rr_data1.csv'
fn2='./hanxue_right0811/sr_label1.csv'
# fn3='./p6_data/after_pre/rl_data1.csv'
# fn4='./p6_data/after_pre/sl_label1.csv'
# fn3='./hanxue_right0530/rr_data1.csv'
# fn4='./hanxue_right0530/sr_label1.csv'

# fn1='./hanxue_right0811/rr_data_split895.csv'  #1161
# fn2='./hanxue_right0811/sr_label_split895.csv'
# fn3='./hanxue_right0530/rr_data1.csv'
# fn4='./hanxue_right0530/sr_label1.csv'

# fn1='./hanxue_right0811/plot_d6.csv'
# fn2='./hanxue_right0811/plot_l6.csv'
# fn3='./hanxue_right0811/test_d6.csv'      
# fn4='./hanxue_right0811/plot_l6.csv' 
# fn1='./hanxue_right0811/rr_data1.csv'
# fn2='./hanxue_right0811/sr_label1.csv'
# fn3='./left1123/rl_data1.csv'
# fn4='./left1123/sl_label1.csv'
fn3='./p15_right0109/rl_data1.csv'
fn4='./p15_right0109/sl_label1.csv'
# fn3='./p6_data/after_pre/rr_data1.csv'
# fn4='./p6_data/after_pre/sr_label1.csv'

fn5='./p15_right0109/kmeans_mxpt_src5.csv'
fn6='./p15_right0109/kmeans_mxpt_tar5.csv'
fn7='./p15_right0109/Ds_pca5.csv'
fn8='./p15_right0109/Dt_pca5.csv'
# fn7='./hanxue_left0825/Ds_pca1.csv'
# fn8='./hanxue_left0825/Dt_pca1.csv'
# fn9='./p10_left1207/train4.csv'
# fn10='./p10_left1207/train_label4.csv'
# fn11='./p10_left1207/test1.csv'

mm='knn' #'knn','logistic','gaussnb','mlp'
kn=10  #假设已知kn个数据的中心
find=0  #find=0:不根据已知中心再找整体中心点； find=1:找整体中心点
kneighbour=1
mxpt=1  #whether adding the max points
mnpt=0
ncomp=24  # left:30,right:24  写入文件的维度
nknn=24  #降维至聚类的维度
data_num=500  #848,895,663
nneighbor=2
# nsplits=1
hand='right'
if hand=='right':
    len_data=17
    label_num=12
    L=['y','u','h','j','n','m','i','k','l','o','p','+']
    # L=['0','1','2']
else:
    len_data=14
    label_num=15
    L=['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
# with open(fn1,"r") as f1:
#     Ds0 = list(csv.reader(f1))
# Ds=np.array(Ds0,dtype=float)
if os.path.exists(fn7):
    remove = open(fn5, "r+")  
    remove.truncate()
    remove = open(fn6, "r+")  
    remove.truncate()
    remove = open(fn7, "r+")  
    remove.truncate()
    remove = open(fn8, "r+")  
    remove.truncate()
    # remove = open(fn9, "r+")  
    # remove.truncate()
    # remove = open(fn10, "r+")  
    # remove.truncate()
def f2norm(B):
    sum=0
    for i in range(len(B)):
        for j in range(len(B[0])):
            sum+=B[i][j]*B[i][j]
    return sum


with open(fn3,"r") as f3:
    Dt = list(csv.reader(f3))
Dt=np.array(Dt,dtype=float)
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
Ds=Ds[0:data_num,:]


Dt0 = pd.read_csv(fn3,header=None)
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
Dt=Dt[0:data_num,:]


with open(fn2,"r") as f2:
    Ls = list(csv.reader(f2))
Ls=np.array(Ls,dtype=object)
Ls=Ls[0:data_num,0]
Ls= [i for ii in Ls for i in ii]
with open(fn4,"r") as f4:
    Lt = list(csv.reader(f4))
Lt=np.array(Lt,dtype=object)
Lt=Lt[0:data_num,0]
Lt= [i for ii in Lt for i in ii]


lda1=PCA(n_components=ncomp)
lda1.fit(Ds,Ls)
Ds30=lda1.transform(Ds)
lda2=PCA(n_components=ncomp)
lda2.fit(Dt,Lt)
Dt30=lda2.transform(Dt)

lda5=PCA(n_components=ncomp)
lda5.fit(Ds,Ls)
Dsn=lda5.transform(Ds)
lda6=PCA(n_components=ncomp)
lda6.fit(Dt,Lt)
Dtn=lda6.transform(Dt)
# with open(fn7,'a+',newline="") as f7:
#     writer=csv.writer(f7)
#     for line in (Ds30):
#         writer.writerow(line)
# with open(fn8,'a+',newline="") as f8:
#     writer=csv.writer(f8)
#     for line in (Dt30):
#         writer.writerow(line)

cen_s1=[]
cen_t1=[]
cen_sn=[]
cen_tn=[]
xkns=[]
ykns=[]
xknt=[]
yknt=[]
train30=[]  #已知标签的数据点，在Ds的基础上加上train30
test30=[]  #删去了已知标签的数据点，作为新的Dt
#在原维度上根据kn个已知数据标签，得到kn个的中心点
for l in L:
    Ds1=[]
    Dt1=[]
    Ds1_30=[]
    Dt1_30=[]
    Ds1_n=[]
    Dt1_n=[]
    mxs=[]
    mxt=[]

    for i in range(len(Ls)):
        if Ls[i]==l:
            Ds1.append(Ds[i,:])
            Ds1_30.append(Ds30[i,:])
            Ds1_n.append(Dsn[i,:])
    for i in range(len(Lt)):
        if Lt[i]==l:
            Dt1.append(Dt[i,:])
            Dt1_30.append(Dt30[i,:])
            Dt1_n.append(Dtn[i,:])
    Ds11=[]
    Dt11=[]
    Ds12=[]
    Dt12=[]
    sets={-1}
    sett={-1}

    while len(sets)<=kn:
        a=random.choice(range(len(Ds1)))
        sets.add(a)
        # Ds11.append(random.choice(Ds1))
        # Ds11=list(set(Ds11[0]))
    for ii in range(kn):
        ykns.append(l)

    while len(sett)<=kn:
        a=random.choice(range(len(Dt1)))
        sett.add(a)
        # Dt11.append(random.choice(Dt1))
        # Dt11=list(set(Dt11[0]))
    for ii in range(kn):
        yknt.append(l)
    sets.discard(-1)
    sett.discard(-1)
    sets=list(sets)
    sett=list(sett)
    for i in sets:
        Ds11.append(Ds1[i])
        Ds12.append(Ds1_n[i])
        xkns.append(Ds1_n[i])
    for i in sett:
        Dt11.append(Dt1[i])
        Dt12.append(Dt1_n[i])
        xknt.append(Dt1_n[i])
        train30.append(Dt1_30[i])
   
    for i in range(len(Dt1_30)):
        h=0
        for j in sett:
            if i==j:
                h=1
                break
        if h==0:
            test30.append(Dt1_30[i])

    Ds11=np.array(Ds11,dtype=float)
    Dt11=np.array(Dt11,dtype=float)
    Ds12=np.array(Ds12,dtype=float)
    Dt12=np.array(Dt12,dtype=float)
    cluster_s1 = KMeans(n_clusters=1,random_state=0).fit(Ds11)
    cen_s1.extend(cluster_s1.cluster_centers_)
    cluster_t1 = KMeans(n_clusters=1,random_state=0).fit(Dt11)
    cen_t1.extend(cluster_t1.cluster_centers_)

    cluster_s12 = KMeans(n_clusters=1,random_state=0).fit(Ds12)
    cen_sn.extend(cluster_s12.cluster_centers_)
    cluster_t12 = KMeans(n_clusters=1,random_state=0).fit(Dt12)
    cen_tn.extend(cluster_t12.cluster_centers_)
cen_s1=list(cen_s1)  #136维
cen_t1=list(cen_t1)
# print('cen_s1,l:',cen_s1,l)
# cen_s30=lda1.transform(cen_s1)
# cen_t30=lda2.transform(cen_t1)
cen_sn=list(cen_sn)
cen_tn=list(cen_tn)

if mm=='knn':
    model1 = KNeighborsClassifier(n_neighbors=nneighbor,weights='distance',metric='manhattan')    #实例化KNN模型
    model2 = KNeighborsClassifier(n_neighbors=nneighbor,weights='distance',metric='manhattan') 
elif mm=='logistic':
    model1 = LogisticRegression()
    model2 = LogisticRegression(C=0.6)
    # model2= LogisticRegression(penalty='l2',dual=False,tol=1e4,C=1.0,fit_intercept=True,
    #                intercept_scaling=1,class_weight=None,random_state=None,solver='liblinear',
    #                max_iter=100,multi_class='ovr',verbose=0,warm_start=False, n_jobs=1)
elif mm=='gaussnb':
    model1=GaussianNB()  #参数不太可调
    model2=GaussianNB()
elif mm=='mlp':
    model1= MLPClassifier()  #hidden_layer_sizes=nerouns,learning_rate_init=lr
    model2= MLPClassifier(hidden_layer_sizes=(50,50),solver='lbfgs')
model1.fit(xkns, ykns)
pres=model1.predict(Dsn)      #放入训练数据进行训练
print('len(pres):',len(pres))           #打印预测内容

# knn2 = KNeighborsClassifier(n_neighbors=nneighbor,weights='distance',metric='manhattan')    
model2.fit(xknt, yknt)
pret=model2.predict(Dtn)     
# print(pret)         

#已知所有标签时，n维下真正的类中心cens_std,cent_std
cens_std=[]
cent_std=[]
for l in L:
    Ds1=[]
    for i in range(len(Ls)):
        if Ls[i]==l:
            Ds1.append(Dsn[i,:])
    cluster_s4 = KMeans(n_clusters=1,random_state=0).fit(Ds1)
    # print('cluster_s4:',cluster_s4)
    cens_std.extend(cluster_s4.cluster_centers_)
for l in L:
    Dt1=[]
    for i in range(len(Lt)):
        if Lt[i]==l:
            Dt1.append(Dtn[i,:])
    cluster_t4 = KMeans(n_clusters=1,random_state=0).fit(Dt1)
    cent_std.extend(cluster_t4.cluster_centers_)
cens_std=list(cens_std)  #nknn维
cent_std=list(cent_std)

#在降维后（30维）上，根据knn得到的分类标签找中心点
if kneighbour==1:
    cen_s4=[]
    cen_t4=[]
    for l in L:
        Ds1=[]
        for i in range(len(pres)):
            if pres[i]==l:
                Ds1.append(Dsn[i,:])
        cluster_s4 = KMeans(n_clusters=1,random_state=0).fit(Ds1)
        # print('cluster_s4:',cluster_s4)
        cen_s4.extend(cluster_s4.cluster_centers_)
    for l in L:
        Dt1=[]
        for i in range(len(pret)):
            if pret[i]==l:
                Dt1.append(Dtn[i,:])
        cluster_t4 = KMeans(n_clusters=1,random_state=0).fit(Dt1)
        cen_t4.extend(cluster_t4.cluster_centers_)
    cen_s4=list(cen_s4)  #nknn维
    cen_t4=list(cen_t4)
    print('len(cen_s4:)',len(cen_s4))

    print(nknn,'维下的距离：')
    print('f2(cen_s1):',f2norm(np.array(cen_sn)))
    print('f2(cen_t1):',f2norm(np.array(cen_tn)))
    print('knn：')
    print('f2(cen_s4-cen_s1):',f2norm(np.array(cen_s4)-np.array(cen_sn)))
    print('f2(cen_t4-cen_t1):',f2norm(np.array(cen_t4)-np.array(cen_tn)))

    print('f2(cens_std):',f2norm(np.array(cens_std)))
    print('f2(cent_std):',f2norm(np.array(cent_std)))
    print('聚类后中心点，未对齐标签：')
    print('f2(cen_s4-cens_std):',f2norm(np.array(cen_s4)-np.array(cens_std)))
    print('f2(cen_t4-cent_std):',f2norm(np.array(cen_t4)-np.array(cent_std)))

if find==1:  #根据整体找中心点
    # lda3=PCA(n_components=30)  #10
    # lda3.fit(Ds,Ls)
    # Ds4=lda3.transform(Ds)
    # lda4=PCA(n_components=30)  #10

    # lda1.fit(Dt,Lt)
    # Dt4=lda1.transform(Dt)
    cen_s30=lda1.transform(cen_s1)
    cen_t30=lda2.transform(cen_t1)

    #寻找距离中心点最近的数据点
    cen_ors=[]  #离聚类中心最近的数据点
    cen_ort=[]
    for cpt in cen_s4:
        mindis=np.dot(np.array(cpt)-np.array(Dsn[0]),np.array(cpt)-np.array(Dsn[0]))
        idxt=0
        for pt in range(1,len(Dsn)):
            tmpdis=np.dot(np.array(cpt)-np.array(Dsn[pt]),np.array(cpt)-np.array(Dsn[pt]))
            if tmpdis<mindis:
                idxt=pt   
                mindis=tmpdis
        print('Ds降维后，距离聚类中心最近的点的索引及标签：id,Ls,pres:',idxt,Ls[idxt],pres[idxt])
        cen_ors.append(Ds30[idxt])
    for cpt in cen_t4:
        mindis=np.dot(np.array(cpt)-np.array(Dtn[0]),np.array(cpt)-np.array(Dtn[0]))
        idxt=0
        for pt in range(1,len(Dtn)):
            tmpdis=np.dot(np.array(cpt)-np.array(Dtn[pt]),np.array(cpt)-np.array(Dtn[pt]))
            if tmpdis<mindis:
                idxt=pt
                mindis=tmpdis
        print('Dt降维后，距离聚类中心最近的点的索引：id,Lt,pret:',idxt,Lt[idxt],pret[idxt])
        cen_ort.append(Dt30[idxt])


    dif_s31=[]
    dif_t31=[]
    for ii in range(label_num):
        ts1=[]
        tt1=[]
        for jj in range(label_num):
            ts1.append(np.dot(np.array(cen_ors[ii])-np.array(cen_s30[jj]),np.array(cen_ors[ii])-np.array(cen_s30[jj])))
            tt1.append(np.dot(np.array(cen_ort[ii])-np.array(cen_t30[jj]),np.array(cen_ort[ii])-np.array(cen_t30[jj])))
        dif_s31.append(ts1)
        dif_t31.append(tt1)
    print('横坐标表示聚类后中心点索引，纵坐标表示真正中心点的索引：')
    print('dif_s31:',np.array(dif_s31).round(2))
    print('dif_t31:',np.array(dif_t31).round(2))

    print('30维下的距离：')
    print('f2(cen_s1):',f2norm(np.array(cen_s30)))
    print('f2(cen_t1):',f2norm(np.array(cen_t30)))
    # print('聚类后中心点，未对齐标签：')
    # print('f2(cen_s2-cen_s1):',f2norm(np.array(cen_s2)-np.array(cen_s30)))
    # print('f2(cen_t2-cen_t1):',f2norm(np.array(cen_t2)-np.array(cen_t30)))
    print('聚类后最近点，未对齐标签：')
    print('f2(cen_ors-cen_s1):',f2norm(np.array(cen_ors)-np.array(cen_s30)))
    print('f2(cen_ort-cen_t1):',f2norm(np.array(cen_ort)-np.array(cen_t30)))


if mxpt==1:  #加入极值点
    mxs=[]
    mxt=[]
    for l in L:
        Ds1=[]
        Dt1=[]
        Dsa=[]
        Dta=[]
        for i in range(len(Ls)):
            if Ls[i]==l:
                Ds1.append(Ds30[i,:])
        for i in range(len(Lt)):
            if Lt[i]==l:
                Dt1.append(Dt30[i,:])

        for i in range(len(Ds1)):
            sm=0
            for j in range(len(Ds1[0])):
                sm+=Ds1[i][j]*Ds1[i][j]
            Dsa.append(sm)
        max_value = max(Dsa) 
        max_idx = Dsa.index(max_value)
        mxs.append(Ds1[max_idx][:])

        for i in range(len(Dt1)):
            sm=0
            for j in range(len(Dt1[0])):
                sm+=Dt1[i][j]*Dt1[i][j]
            Dta.append(sm)
        max_value = max(Dta) 
        max_idx = Dta.index(max_value)
        mxt.append(Dt1[max_idx])

if mnpt==1:
    mns=[]
    mnt=[]
    for l in L:
        Ds1=[]
        Dt1=[]
        Dsa=[]
        Dta=[]
        for i in range(len(Ls)):
            if Ls[i]==l:
                Ds1.append(Ds2[i,:])
        for i in range(len(Lt)):
            if Lt[i]==l:
                Dt1.append(Dt2[i,:])

        for i in range(len(Ds1)):
            sm=0
            for j in range(len(Ds1[0])):
                sm+=Ds1[i][j]*Ds1[i][j]
            Dsa.append(sm)
        min_value = min(Dsa) 
        min_idx = Dsa.index(min_value)
        mns.append(Ds1[min_idx][:])

        for i in range(len(Dt1)):
            sm=0
            for j in range(len(Dt1[0])):
                sm+=Dt1[i][j]*Dt1[i][j]
            Dta.append(sm)
        min_value = min(Dta) 
        min_idx = Dta.index(min_value)
        mnt.append(Dt1[min_idx])

if find==1:
    with open(fn5,'a+',newline="") as f5:
        writer=csv.writer(f5)
        for line in (cen_ors):  #cen_s2
            writer.writerow(line)
    with open(fn6,'a+',newline="") as f6:
        writer=csv.writer(f6)
        for line in (cen_ort):   #cen_t2
            writer.writerow(line)
elif kneighbour==1 and find==0:
    with open(fn5,'a+',newline="") as f5:
        writer=csv.writer(f5)
        for line in (cens_std):  #cen_ors，cen_s4
            writer.writerow(line)
    with open(fn6,'a+',newline="") as f6:
        writer=csv.writer(f6)
        for line in (cen_t4):   #cen_ort,cen_t4
            writer.writerow(line)
else:
    with open(fn5,'a+',newline="") as f5:
        writer=csv.writer(f5)
        for line in (cen_s30):
            writer.writerow(line)
    with open(fn6,'a+',newline="") as f6:
        writer=csv.writer(f6)
        for line in (cen_t30):
            writer.writerow(line)
if mxpt==1:
    with open(fn5,'a+',newline="") as f5:
        writer=csv.writer(f5)
        for line in (mxs):
            writer.writerow(line)
    with open(fn6,'a+',newline="") as f6:
        writer=csv.writer(f6)
        for line in (mxt):
            writer.writerow(line)     
if mnpt==1:
    with open(fn5,'a+',newline="") as f5:
        writer=csv.writer(f5)
        for line in (mns):
            writer.writerow(line)
    with open(fn6,'a+',newline="") as f6:
        writer=csv.writer(f6)
        for line in (mnt):
            writer.writerow(line)    

with open(fn7,'a+',newline="") as f7:
    writer=csv.writer(f7)
    for line in (Ds30):
        writer.writerow(line)
with open(fn8,'a+',newline="") as f8:
    writer=csv.writer(f8)
    for line in (Dt30):
        writer.writerow(line)

# with open(fn9,'a+',newline="") as f9:
#     writer=csv.writer(f9)
#     for line in (train30):
#         writer.writerow(line)
# with open(fn10,'a+',newline="") as f10:
#     writer=csv.writer(f10)
#     for line in (yknt):
#         writer.writerow(line)
# with open(fn11,'a+',newline="") as f11:
#     writer=csv.writer(f11)
#     for line in (test30):
#         writer.writerow(line)