import numpy as np
import math
import pywt
import csv

def sgn(num):
    if (num > 0):
        return 1.0
    elif (num == 0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    data = new_df
    # data = data.values.T.tolist()  # convert np.ndarray() to list
    data=list(data)
    w = pywt.Wavelet('sym8')
    # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # wave decomposition

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca5) 

    #soft-and-hard threshold
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


def denoise(data):
    data_denoising = wavelet_noising(data)
    return data_denoising


hand='right'

fn1='./data/data_new right.csv'
fn2='./data/wavelet_origin_right.csv'  

with open(fn1,"r") as f1:
    data1 = list(csv.reader(f1))
data1=np.array(data1)
if hand=='left':
    s_data=data1[:,13:21]   #lefthand
else:
    s_data=data1[:,0:8]   #righthand
n=len(s_data)
print('length data:',n)

h_data=s_data[:,:].astype(np.int16)


for i in range(8):
    s1=s_data[:,i].astype(np.int16)
    h11=denoise(s1)
    for j in range(len(h11)):
            h_data[j,i]=h11[j]
            
data=h_data
with open(fn2,'a+',newline="") as f30:
    writer=csv.writer(f30)
    for line in data:
        writer.writerow(line)   
