#整体归一化
import numpy as np
import csv
hand='left'
num='withnum'  #不带数字
#文件路径
fn1='processed_data/cl_data.csv'
fn2='processed_data/test_data.csv'

def main():
    global hand
    global num
    
    with open(fn1,"r") as f1:
        data = list(csv.reader(f1))
    data=np.array(data,dtype=float)
    n=len(data)
    if hand=='right':
        if num=='nonum':
            maxd=[2437.4, 5030.72, 3819.7, 2403.1, 4172.7, 1793.0, 2464.6, 2015.45]
            mind=[823.48, 1309.5, 1427.0, 1318.0, 1925.0, 1006.14, 1045.0, 985.61]
        else:
            maxd=[2437.4, 5030.72, 4513.31, 2477.35, 4172.7, 1822.12, 2639.29, 2102.0]
            mind=[823.48, 1106.64, 1427.0, 1193.85, 1836.0, 895.06, 1035.85, 960.0]

    else:  #left
        if num=='nonum':  #不带数字
            maxd=[988.78, 1132.29, 870.0, 948.84, 875.0, 771.0, 965.86, 923.68]
            mind=[652.0, 748.9, 697.0, 700.87, 725.0, 638.95, 718.0, 627.88]
        else:
            maxd=[988.78, 1132.29, 872.61, 948.84, 891.18, 771.0, 991.83, 923.68]
            mind=[647.0, 748.9, 697.0, 700.87, 725.0, 638.95, 718.0, 627.88]
    d=[]

    # for i in range(8):
    #     maxd.append(max(data[:,i]))
    #     mind.append(min(data[:,i]))
    for i in range(n):
        line=[]
        for j in range(8):
            line.append((data[i,j]-mind[j])/(maxd[j]-mind[j]))
        d.append(line)
    with open(fn2,'w+',newline="") as f2:
        writer=csv.writer(f2)
        for line in d:
                writer.writerow(line)
                
if __name__ == '__main__':
    main()