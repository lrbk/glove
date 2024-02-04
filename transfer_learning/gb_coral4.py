from numpy import mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import csv

pca_pred=1
hand='right'
num='nonum'
coral=1
type='coral'
mm='logistic'  # Select the model: 'mlp' or 'gaussnb' or 'knn' or 'logistic'


# data_path=os.path.join('datas',f"{hand}_{num}_data.csv")
# label_path=os.path.join('datas',f"{hand}_{num}_label.csv")
# lda_path = os.path.join('pretrained_model', f"lda_{hand}_{num}.model")
# model_path = os.path.join('pretrained_model', f"model_{hand}_{num}.model")
# svm_path = os.path.join('pretrained_model', f"svm_{hand}_{num}.model")

if hand=='left':
    if coral==0:
        data_path='../data/rl_data1.csv'
        label_path='../data/sl_label1.csv'
    if coral==1:
        data_path='./p1_left/Ds_pca27.csv'
        test_d='./p1_left/Dt_pca27.csv'
        label_path='../data/sl_label1.csv'
        test_l='./p1_left/sl_label1.csv'
        fna='./p1_left/A_nolabel/A22_5.csv'
        re=0  #re=1:reverse Ds and Dt
        ncom=30
        data_num=500  
    lda_path = "..data/model/lda_left_nonum.model"
    model_path = "..data/model/model_left_nonum.model"
    svm_path = "..data/model/svm_left_nonum.model"
else:
    if coral==0:
        data_path='..data/rr_data1.csv'
        label_path='..data/sr_label1.csv'
        lda_path = "..data/model/lda_right_nonum.model"
        model_path = "..data/model/model_right_nonum.model"
        svm_path = "..data/model/svm_right_nonum.model"
    if coral==1:
        ###降维后
        data_path='./p1_right/Ds_pca5.csv' 
        test_l='./p1_right/sl_label1.csv'
        test_d='./p1_right/Dt_pca5.csv'
        label_path='..data/sr_label1.csv'
        fna='./p1_right/A_nolabel/A3_5.csv'
        re=0
        ncom=24
        data_num=500
        lda_path = "..data/model/lda_right_nonum_coral.model"
        model_path = "..data/model/model_right_nonum_coral.model"
        svm_path = "..data/model/svm_right_nonum_coral.model"

#set the error pairs and the labels
if hand=='right':
    if num=='nonum':
        # c_err=[('m','n'),('n','h'),('y','u'),('h','j'),('n','+')] 
        L=['y','u','h','j','n','m','i','k','l','o','p','+']
        c_err=[]
    else:
        c_err=[('o','9'),('0','p'),('m','n'),('y','u','h'),('y','u'),('h','j'),('y','6'),('8','i')]
        L=['6','7','8','9','0','y','u','h','j','n','m','i','k','l','o','p','+']
else:
    if num=='nonum':
        c_err=[('t','g'),('g','b'),('g','f'),('f','b'),('f','d')]
        L=['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
    else:
        c_err=[] 
        L=['1','2','3','4','5','q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']

#set the parameters(hand,the min dim of LDA,the length of each wave)
if hand=='left':  
    min_dim=10
    len_data=14   
else:
    min_dim=9  #9
    len_data=17
max_dim=min_dim+1
err=0.01  #the error rate of the error pairs
n_datas=15  #the times of data augmentation
nsplits=5  #the times of cross validation
nerouns=(25) #the nerouns of mlp
lr=0.3  #the learning rate of mlp


# define dataset
y = pd.read_csv(label_path,header=None)#The training labels
numy=len(y)
y=y.values.ravel()

one_hot = LabelBinarizer()   
L0=one_hot.fit(L)


print('datapath:',data_path)
if coral==0:
    data = pd.read_csv(data_path,header=None)#Data gained from Normalized data
    X=[]
    for i in range(0,numy):
        x=[]
        for j in range(0,8):#We have 8 sensors
            height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
            x.extend(height)
        X.append(x)

    times=0
    for dim in range(min_dim,max_dim):
        print('Dim',dim)
        train_score=[]
        test_score=[]
        train_svm_score=[]
        test_svm_score=[]
        tr_or=[]
        te_or=[]
        LDA_data=[]
        gt_labels=[]
        gb_labels=[]
        pred_labels=[]
        if coral==0:
            kf = KFold(n_splits=nsplits)
            for train,test in kf.split(X):
                print('times:',times)
                times+=1
                Xtrain0=[]
                xtest0=[]
                Ytrain=[]
                ytest=[]
                Ytrain0=[]
                ytest0=[]
                for val in train:
                    Xtrain0.append(X[val])
                    Ytrain.append(y[val])
                    # Ytrain0.append(y0[val])
                for val in test:
                    xtest0.append(X[val])
                    ytest.append(y[val])   


                #LDA to dimension reduction
                lda=LDA(n_components=dim)
                lda.fit(Xtrain0,Ytrain)
                Xtrain=lda.transform(Xtrain0)
                xtest=lda.transform(xtest0)
                LDA_data.extend(xtest)

                df = pd.DataFrame(data=Xtrain, columns=[f'LDA_Component_{i+1}' for i in range(Xtrain.shape[1])])
                df.insert(0, 'Label', Ytrain)

                # save data with labels
                df.to_csv('data_with_labels.csv', index=False)
                
                # save LDA model
                with open(lda_path, 'wb') as f:
                    pickle.dump(lda, f)

                xtr_or=Xtrain
                ytr_or=Ytrain

                n_or=len(ytr_or)
                n=len(Ytrain)
                data=dict()
                for label in L:
                    label_set=[]
                    for ii in range(n):
                        if Ytrain[ii]==label:
                            label_set.append(Xtrain[ii])
                    data[label]=label_set

                Xtrain=[]
                Ytrain=[]
                for i in range(len(L)):
                    label=L[i]
                    mean=np.mean(data[label],axis=0)
                    cov=np.cov(np.transpose(data[label]))
                    n_or=len(data[label])
                    new_train = np.random.multivariate_normal(mean, cov, size=(n_datas*n_or))
                    data[label]=np.vstack((data[label],new_train))
                    n=len(data[label])
                    if label==L[0]:
                        Xtrain=data[label]
                    else:
                        Xtrain=np.vstack((Xtrain,data[label]))
                    for i in range(n):
                        Ytrain.append(label)

                #shuffle the data
                ii=[i for i in range(len(Xtrain))]
                random.shuffle(ii)
                xx=[]
                yy=[]
                for i in ii:
                    xx.append(Xtrain[i,:])
                    yy.append(Ytrain[i])
                xx=np.array(xx,dtype=float)

                # select the model used in the first layer
                if mm=='mlp':
                    model = MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)  
                    model1= MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)
                if mm=='gaussnb':
                    model=GaussianNB()
                    model1=GaussianNB()
                if mm == 'knn':
                    model = KNeighborsClassifier(n_neighbors=3)
                    model1 = KNeighborsClassifier(n_neighbors=3)
                if mm == 'logistic':
                    model = LogisticRegression(max_iter=300)
                    model1 = LogisticRegression(max_iter=300)
                    
                # transform the labels to one-hot if necessary
                if mm=='mlp':
                    #mlp, need one-hot
                    y0=one_hot.transform(yy)
                    y0test=one_hot.transform(ytest)
                    y0tr_or=one_hot.transform(ytr_or)
                else:
                    #other models, no need one-hot
                    y0=yy
                    y0test=ytest
                    y0tr_or=ytr_or  

                model.fit(xx,y0)  # training process
                model1.fit(xtr_or,y0tr_or)

                pre = model.predict(xtest)  # predict process 
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                gb_labels.extend(pre)
                score=model.score(xx,y0)  
                train_score.append(score)
                score=model.score(xtest, y0test) 
                test_score.append(score)
                score=model1.score(xtr_or,y0tr_or) 
                pre_train=model1.predict(xtr_or)
                pre_test=pre
                gt_labels.extend(y0test)  
                tr_or.append(score)
                score=model1.score(xtest, y0test)
                te_or.append(score)    


                #The Second Layer: SVM
                id_svm={}
                for j in range(len(y0tr_or)):
                    for i in range(len(c_err)):
                        if y0tr_or[j] in c_err[i]:
                            ids=id_svm.get(c_err[i],[])
                            ids.append(j)
                            id_svm[c_err[i]]=ids  

                xtr_svm=[]
                ytr_svm=[]
                for k in range(len(c_err)):
                    for key,value in id_svm.items():
                        if key==c_err[k]:
                            data_svm=[]
                            lb_svm=[]
                            for v in value:
                                data_svm.append(Xtrain0[v])  #Xtrain0: no data augumentation, no dimension reduction
                                lb_svm.append(y0tr_or[v])
                            xtr_svm.append(data_svm)
                            ytr_svm.append(lb_svm)


                svm_model={}
                for k in range(len(c_err)):
                    svm_model[c_err[k]]=LinearSVC()
                    svm_model[c_err[k]].fit(xtr_svm[k],ytr_svm[k])

                with open(svm_path, 'wb') as f:
                    pickle.dump(svm_model, f)

                #Using svm to correct the error
                pre_svm_test=pre_test
                pre_svm_train=pre_train
                for k in range(len(c_err)):
                    for i in range(len(pre_test)):
                        if pre_test[i] in c_err[k]:
                            pre_svm_test[i]=(svm_model[c_err[k]].predict([xtest0[i]]))[0]
                for k in range(len(c_err)):
                    for i in range(len(pre_train)):
                        if pre_train[i] in c_err[k]:
                            pre_svm_train[i]=(svm_model[c_err[k]].predict([Xtrain0[i]]))[0]
                
                pred_labels.extend(pre_svm_test)

                #calculate the accuracy
                count=0
                for i in range(len(y0tr_or)):
                    if pre_svm_train[i]==y0tr_or[i]:
                        count+=1
                train_svm_score.append(count/len(y0tr_or))
                print('train_svm_score:',count/len(y0tr_or))
                count=0
                for i in range(len(y0test)):
                    if pre_svm_test[i]==y0test[i]:
                        count+=1
                test_svm_score.append(count/len(y0test))
                print('test_svm_score:',count/len(y0test))
                
            print('after data augumentation:(augumentation times%d)' %n_datas)
            print('mean train score:',sum(train_score)/nsplits)
            print('mean test score:',sum(test_score)/nsplits)
            print('before data augumentation:' )
            print('mean train score:',sum(tr_or)/nsplits)
            print('mean test score:',sum(te_or)/nsplits)  
            print('no data augumentation, after svm:')
            print('mean train score:',sum(train_svm_score)/nsplits)
            print('mean test score:',sum(test_svm_score)/nsplits)      
            
            #plot the confusion matrix
            # confusion_mat = confusion_matrix(gt_labels, pred_labels, labels=L, normalize='true')
            # confusion_mat = confusion_mat.astype(np.float64)
            # c_percent = confusion_mat / confusion_mat.sum(axis=1, keepdims=True) * 100
            # c_percent=c_percent.astype(np.int_)
            # disp = ConfusionMatrixDisplay(confusion_matrix=c_percent, display_labels=L)
            # disp.plot(
            #     include_values=True,            
            #     cmap="viridis",                 
            #     ax=None,                        
            #     xticks_rotation="horizontal",  
            #     values_format="d"               
            # )
            # plt.title(f'{hand} SVM Confusion Matrix')
            # plt.show()

#######################################################################################################################
# coral   
    else:
        with open(label_path,"r") as f2:
            Ytrain1 = list(csv.reader(f2))
        Ytrain = [i for ii in Ytrain1 for i in ii]
        Ytrain=np.array(Ytrain,dtype=object)
        Ytrain=Ytrain[0:data_num]

        with open(test_l,"r") as f4:
            ytest = list(csv.reader(f4))
        ytest1=np.array(ytest,dtype=object)
        ytest = [i for ii in ytest1 for i in ii]
        ytest=np.array(ytest,dtype=object)
        ytest=ytest[0:data_num]
        
        if pca_pred==1:
            with open(data_path,"r") as f1:
                Xtrain0 = list(csv.reader(f1))
            Xtrain0=np.array(Xtrain0,dtype=float)
            
            if re==0: 
                with open(test_d,"r") as f3:
                    xtest0 = list(csv.reader(f3))
                xtest0=np.array(xtest0,dtype=float)
            else:  
                data = pd.read_csv(test_d,header=None)#Data gained from Normalized data
                xtest0=[]
                for i in range(0,len(ytest)):
                    x=[]
                    for j in range(0,8):#We have 8 sensors
                        height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
                        x.extend(height)
                    xtest0.append(x)
                xtest0=np.array(xtest0,dtype=float)

                lda0=PCA(n_components=ncom)
                lda0.fit(xtest0[0:data_num][:],ytest)
                xtest0=lda0.transform(xtest0)
        else:
            data = pd.read_csv(data_path,header=None)#Data gained from Normalized data
            Xtrain0=[]
            for i in range(0,numy):
                x=[]
                for j in range(0,8):#We have 8 sensors
                    height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
                    x.extend(height)
                Xtrain0.append(x)
            Xtrain0=np.array(Xtrain0,dtype=float)

            data = pd.read_csv(test_d,header=None)#Data gained from Normalized data
            xtest0=[]
            for i in range(0,len(ytest)):
                x=[]
                for j in range(0,8):#We have 8 sensors
                    height=data.iloc[len_data*i:len_data*i+len_data,j].to_list()
                    x.extend(height)
                xtest0.append(x)
            xtest0=np.array(xtest0,dtype=float)


        times=0
        for dim in range(min_dim,max_dim):
            print('Dim',dim)
            train_score=[]
            test_score=[]
            train_svm_score=[]
            test_svm_score=[]
            tr_or=[]
            te_or=[]
            LDA_data=[]
            gt_labels=[]
            gb_labels=[]
            pred_labels=[]
            Xt=xtest0
            yt=ytest
            #de-mean Ds, Dt
            means=[]
            meant=[]
            stds=[]
            stdt=[]
            for j in range(len(Xtrain0[0])):
                means.append(np.mean(Xtrain0[:,j]))
                meant.append(np.mean(Xt[:,j]))
                stds.append(np.std(Xtrain0[:,j]))
                stdt.append(np.std(Xtrain0[:,j]))
                for i in range(len(Xtrain0)):
                    Xtrain0[i,j]=(Xtrain0[i,j]-means[j])
                for i in range(len(Xt)):
                    Xt[i,j]=(Xt[i,j]-meant[j])
            xtest0=Xt

            
            for i in range(len(Ytrain)):
                for j in range(len(L)):
                    if Ytrain[i]==L[j]:
                        Ytrain[i]=j
                        continue
            for i in range(len(ytest)):
                for j in range(len(L)):
                    if ytest[i]==L[j]:
                        ytest[i]=j
                        continue
            Ytrain=np.array(Ytrain,dtype=int)
            ytest=np.array(ytest,dtype=int)
            yt=ytest

            if type=='coral':
                with open(fna,"r") as f9:
                    A = list(csv.reader(f9))
                A=np.array(A,dtype=float)
                Xtrain0=np.dot(Xtrain0,A)

            if pca_pred==0:
                #LDA to dimension reduction
                lda=LDA(n_components=dim)
                lda.fit(Xtrain0,Ytrain)
                Xtrain=lda.transform(Xtrain0)
                xtest=lda.transform(xtest0)
                LDA_data.extend(xtest)

                df = pd.DataFrame(data=Xtrain, columns=[f'LDA_Component_{i+1}' for i in range(Xtrain.shape[1])])
                df.insert(0, 'Label', Ytrain)

                # save data with labels
                df.to_csv('data_with_labels.csv', index=False)
                
                # save LDA model
                with open(lda_path, 'wb') as f:
                    pickle.dump(lda, f)

            else:
                Xtrain=Xtrain0
                xtest=xtest0


            xtr_or=Xtrain  
            ytr_or=Ytrain

            n_or=len(ytr_or)
            n=len(Ytrain)
            data=dict()
            for label in range(len(L)):
                label_set=[]
                for ii in range(n):
                    if Ytrain[ii]==label:
                        label_set.append(Xtrain[ii])
                data[label]=label_set

            Xtrain=[]  #after dimension reduction
            Ytrain=[]
            for i in range(len(L)):
                label=i
                mean=np.mean(data[label],axis=0)
                cov=np.cov(np.transpose(data[label]))
                n_or=len(data[label])
                new_train = np.random.multivariate_normal(mean, cov, size=(n_datas*n_or))
                data[label]=np.vstack((data[label],new_train))
                n=len(data[label])
                if label==0:
                    Xtrain=data[label]
                else:
                    Xtrain=np.vstack((Xtrain,data[label]))
                for i in range(n):
                    Ytrain.append(label)

            #shuffle the data
            ii=[i for i in range(len(Xtrain))]
            random.shuffle(ii)
            xx=[]
            yy=[]
            for i in ii:
                xx.append(Xtrain[i,:])
                yy.append(Ytrain[i])
            xx=np.array(xx,dtype=float)

            # select the model used in the first layer
            if mm=='mlp':
                model = MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)  
                model1= MLPClassifier(hidden_layer_sizes=nerouns,learning_rate_init=lr)
            if mm=='gaussnb':
                model=GaussianNB()
                model1=GaussianNB()
            if mm == 'knn':
                model = KNeighborsClassifier(n_neighbors=3)
                model1 = KNeighborsClassifier(n_neighbors=3)
            if mm == 'logistic':
                model = LogisticRegression(max_iter=300)
                model1 = LogisticRegression(max_iter=300)
                
            # transform the labels to one-hot if necessary
            # if mm=='mlp':
            #     #mlp, need one-hot
            #     y0=one_hot.transform(yy)
            #     y0test=one_hot.transform(ytest)
            #     y0tr_or=one_hot.transform(ytr_or)
            # else:
            #other models, no need one-hot
            y0=yy
            y0test=ytest
            y0tr_or=ytr_or  

            model.fit(xx,y0)  # training process
            model1.fit(xtr_or,y0tr_or)

            pre = model.predict(xtest)  # predict process 
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            gb_labels.extend(pre)
            score=model.score(xx,y0)  
            train_score.append(score)
            print('after dimension reduction,train_score:',score)
            score=model.score(xtest, y0test) 
            test_score.append(score)
            print('after dimension reduction,test_score:',score)
            score=model1.score(xtr_or,y0tr_or) 
            pre_train=model1.predict(xtr_or)
            pre_test=pre
            gt_labels.extend(y0test)  
            tr_or.append(score)
            print('no dimension reduction,train_score:',score)
            score=model1.score(xtest, y0test)
            te_or.append(score)    
            print('no dimension reduction,test_score:',score)


            #The Second Layer: SVM
            id_svm={}
            for j in range(len(y0tr_or)):
                for i in range(len(c_err)):
                    if y0tr_or[j] in c_err[i]:
                        ids=id_svm.get(c_err[i],[])
                        ids.append(j)
                        id_svm[c_err[i]]=ids  

            xtr_svm=[]
            ytr_svm=[]
            for k in range(len(c_err)):
                for key,value in id_svm.items():
                    if key==c_err[k]:
                        data_svm=[]
                        lb_svm=[]
                        for v in value:
                            data_svm.append(Xtrain0[v])  #Xtrain0: no dimension reduction, no data augumentation
                            lb_svm.append(y0tr_or[v])
                        xtr_svm.append(data_svm)
                        ytr_svm.append(lb_svm)


            svm_model={}
            for k in range(len(c_err)):
                svm_model[c_err[k]]=LinearSVC()
                svm_model[c_err[k]].fit(xtr_svm[k],ytr_svm[k])

            with open(svm_path, 'wb') as f:
                pickle.dump(svm_model, f)

            #Using svm to correct the error
            pre_svm_test=pre_test
            pre_svm_train=pre_train
            for k in range(len(c_err)):
                for i in range(len(pre_test)):
                    if pre_test[i] in c_err[k]:
                        pre_svm_test[i]=(svm_model[c_err[k]].predict([xtest0[i]]))[0]
            for k in range(len(c_err)):
                for i in range(len(pre_train)):
                    if pre_train[i] in c_err[k]:
                        pre_svm_train[i]=(svm_model[c_err[k]].predict([Xtrain0[i]]))[0]
            
            pred_labels.extend(pre_svm_test)

            #calculate the accuracy
            count=0
            for i in range(len(y0tr_or)):
                if pre_svm_train[i]==y0tr_or[i]:
                    count+=1
            train_svm_score.append(count/len(y0tr_or))
            print('train_svm_score:',count/len(y0tr_or))
            count=0
            for i in range(len(y0test)):
                if pre_svm_test[i]==y0test[i]:
                    count+=1
            test_svm_score.append(count/len(y0test))
            print('test_svm_score:',count/len(y0test))
            
        
        #plot the confusion matrix
        # confusion_mat = confusion_matrix(gt_labels, pred_labels, labels=L, normalize='true')
        # confusion_mat = confusion_mat.astype(np.float64)
        # c_percent = confusion_mat / confusion_mat.sum(axis=1, keepdims=True) * 100
        # c_percent=c_percent.astype(np.int_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=c_percent, display_labels=L)
        # disp.plot(
        #     include_values=True,            
        #     cmap="viridis",                 
        #     ax=None,                        
        #     xticks_rotation="horizontal",  
        #     values_format="d"               
        # )
        # plt.title(f'{hand} SVM Confusion Matrix')
        # plt.show()
