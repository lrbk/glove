#global normalization
import numpy as np
import csv

#file path
fn1='./data/cr_data1.csv'
fn2='./data/rr_data1.csv'
with open(fn1,"r") as f1:
    data = list(csv.reader(f1))
data=np.array(data,dtype=float)
n=len(data)
print('length:',n)
maxd=[]
mind=[]
d=[]
for i in range(8):
    maxd.append(max(data[:,i]))
    mind.append(min(data[:,i]))
for i in range(n):
    line=[]
    for j in range(8):
        line.append((data[i,j]-mind[j])/(maxd[j]-mind[j]))
    d.append(line)
with open(fn2,'a+',newline="") as f2:
    writer=csv.writer(f2)
    for line in d:
            writer.writerow(line)