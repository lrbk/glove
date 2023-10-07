import csv
import numpy as np
filename1='data_new.csv'
filename4='data/sl_data.csv'  #save data
#Firstly we need to run the program 
nn=100 #length of data
len_window=4  #average window length
hand='right'



def cums(n1,da,h0,h1,hand):  #n1:length of window; da:data; h0:threshold of wave; h1:threshold of wave start and end
    nn=len(da)
    s_index={}
    e_index={}
    wave_index=[]
    maxd=[]
    if hand=='left':
        mnl=500
        mxl=1500
    else:
        mnl=600
        mxl=8000
    
    for k in range(8):
        avg=[]
        davg=[]
        l0=da[:,k]
        minl=min(l0)
        maxl=max(l0)
        if minl<mnl or maxl>mxl:
            return -1

        for i in range(nn-n1):
            avg.append(round(sum(l0[i:i+n1-1])/n1,2))
            if i>0:
                davg.append(round(avg[i]-avg[i-1],2))
                if abs(avg[i]-avg[i-1])>h0:
                    s_index[i-1] = s_index.get(i-1, 0) + 1
                    wave_index.append(i-1)
        maxd.append(max(davg))
    if len(wave_index)==0:
        return -4
    if max(wave_index)-min(wave_index)>50:
        return -5

    endkey=[]
    for key,value in s_index.items():
        if value==max(s_index.values()):
            skey=key
            break
    for k in range(8):
        avg=[]
        l0=da[:,k]
        for i in range(max(0,skey-15-n1),min(skey+20-n1,nn-n1)):
            j=i-max(0,skey-15-n1)
            avg.append(round(sum(l0[i:i+n1-1])/n1,2))
            if j>0:
                if abs(avg[j]-avg[j-1])>h1:
                   endkey.append(i-1)  


    s=min(endkey)+n1-1
    e=max(endkey)+n1-1

    if hand=='left':
        aa=4
        bb=25
    else:  #right
        aa=10
        bb=45
    # if 4<e-s<25:
    if aa<e-s<bb:  
        sens=da[s:e,:]
        global sum_wave_len
        sum_wave_len+=e-s+1
        with open(filename4,'w+',newline="") as f4:
            writer=csv.writer(f4)
            for line in sens:
                    writer.writerow(line)
            writer.writerow('\n')
        return 1
    else:
        print('s,e,e-s:',s,e,e-s)
        return -2

def main():
    global hand
    if hand=='left':
        h0=4 
        h1=2
    else:
        h0=30
        h1=20
    
    
    with open(filename1,"r") as f1:
        data1 = list(csv.reader(f1))
    data1=np.array(data1)
    if hand=='left':
        data=data1[0:100,13:21]   #righthand: data=data1[0:100,0:8] ;lefthand: data=data1[0:100,13:21] 
    else:
        data=data1[0:100,0:8]
    n=len(data)
    de=data
    k_label=[] 
    global sum_wave_len
    sum_wave_len=0
    k_label.append(cums(len_window,data.astype(np.int16),h0,h1,hand))  
 

    print('-1,',k_label.count(-1))
    print('-2,',k_label.count(-2))
    print('-5,',k_label.count(-5))
    print('-4,',k_label.count(-4))
    print('1,',k_label.count(1))
    print('avg wave length',sum_wave_len)

if __name__ == '__main__':
    main()




