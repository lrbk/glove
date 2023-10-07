import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

filename1='processed_data/sl_data.csv'  #the data after deal_22,which is not regularized
filename2='processed_data/cl_data.csv' #the data after regularizied
hand='left'
#regular the data in different length to the same length

def reg(da,l):
    da=np.array(da)
    l1=len(da)
    de=[]
    x=[val for val in range(l1)]
    x_new=[]
    for i in range(l):         
        x_new.append(i/(l-1)*(l1-1))
    for j in range(8):
        line=[]
        line1=[]
        sen=da[:,j].astype(np.int_)
        line.append(sen[0])
        f5=CubicSpline(x,sen)
        line1=np.round(f5(x_new),2)
        de.append(line1)
   
    de=np.array(de)  
    de=de.T
    with open(filename2,'w+',newline="") as f2:
        writer=csv.writer(f2)
        for lin in de:
                writer.writerow(lin)

def main():
    global hand
    if hand=='left':
        l=15
    else:
        l=20
    with open(filename1,"r") as f1:
        data = list(csv.reader(f1))
    data=np.array(data,dtype=object)
    n=len(data)
    i=0
    h=['\n']
    lth=[]
    while 1:
        da=[]
        if i>=n:
            break
        while 1:
            if data[i]!=h:
                da.append(data[i])
                i+=1
            else:
                i+=1
                break
        lth.append(len(da))

        if len(da)==l:
            with open(filename2,'w+',newline="") as f2:
                writer=csv.writer(f2)
                for line in da:
                        writer.writerow(line)
        else:
            reg(da,l)

if __name__ == '__main__':
    main()



            
