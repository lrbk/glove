import os
import warnings

import serial
import serial.tools.list_ports
import numpy as np
from time import sleep
import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from math import sqrt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
# from scipy import signal
import math
import pywt
import pyttsx3


class glv():
    def __init__(self):
        self.h0_l=13
        self.h1_l=9.5
        self.h0_r=4
        self.h1_r=2
        self.hand='left'
        self.num='nonum'
        self.nn=50
        self.run=1
        self.lw=4  #len_window
        self.fn1='./data_new.csv'
        self.fn2='./test_origin.csv'
        self.fn3='./test_oriwave.csv'
        self.fn6='./max_min.csv'
        self.fn21='./test_origin_left.csv'
        self.fn22='./test_origin_right.csv'
        self.fn3='./test_lowpass.csv'
        self.fn4='./test2origin.csv'
        self.label=[]
        self.sum_wave_len=0
        self.times=0
        self.max_l=[0,0,0,0,0,0,0,0]
        self.min_l=[6000,6000,6000,6000,6000,6000,6000,6000]
        self.max_r=[0,0,0,0,0,0,0,0]
        self.min_r=[6000,6000,6000,6000,6000,6000,6000,6000]
        self.wl_l=14
        self.wl_r=17
        self.ii=1
        self.wave=0
        self.tr=0  #未连接手套时的模拟程序
        self.out=[]



        self.root=tk.Tk()

        self.root.title("Gesture Recognition GUI")
        self.root.attributes('-fullscreen', True)
        root_width = self.root.winfo_screenwidth()
        root_height = self.root.winfo_screenheight()
        self.root.bind("<Escape>", self.exit_app)

        # 设置标题字体
        title_font = ("Arial", 20, "bold")

        # 标题
        title_label = tk.Label(self.root, text="S.M.A.R.T LAB", font=title_font)
        title_label.pack(anchor=tk.NW, padx=20, pady=20)  # 放置在左上角

        # # 加载 PNG 图片并缩放
        image = Image.open("logo.png")  # 替换为您的 PNG 图片文件路径
        new_width = 150  # 新的宽度
        new_height = 150  # 根据宽度比例计算新的高度
        image = image.resize((new_width, new_height))
        photo = ImageTk.PhotoImage(image)

        # 创建 Label 来显示缩小后的图片
        image_label = tk.Label(self.root, image=photo)
        image_label.photo = photo  # 保留对图片的引用，否则图片会被垃圾回收
        image_label.place(x=root_width - new_width-10, y=0)  # 将图片放置在右上角

        # 创建predict_frame
        self.predict_frame = tk.Frame(self.root,width=root_width*0.8, height=root_height*0.3)
        self.predict_frame.pack(side=tk.TOP, padx=20, pady=20,anchor=tk.N)

        # 创建predict_text并放置在predict_frame中
        self.final_predict_text = tk.Text(self.predict_frame, wrap=tk.WORD, font=("Arial", 16) ,height=16,width=100)
        self.final_predict_text.pack(fill=tk.BOTH, expand=True)  # 填充整个predict_frame


        # 创建图形绘制区域plot_frame
        self.plot_frame = tk.Frame(self.root, width=root_width * 0.2)
        self.plot_frame.pack(side=tk.LEFT, expand=True,padx=0, pady=0, anchor=tk.SW)  # 放置在左侧，占据左侧底部 40% 的区域

        # 创建图像
        self.figure_sensor = Figure(figsize=(6, 3), dpi=100)
        self.plot_sensor = self.figure_sensor.add_subplot(111)

        self.figure_sensor.subplots_adjust(top=0.9,bottom=0.15,left=0.15, right=0.85)
        self.canvas_sensor = FigureCanvasTkAgg(self.figure_sensor, master=self.plot_frame)
        self.canvas_sensor.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,  expand=True)
        # self.plot_sensor.legend(column_labels,loc='upper left', bbox_to_anchor=(1, 1))  # 添加图例，并设置位置在图像外的右上角

        # 创建predict_frame
        self.R_frame = tk.Frame(self.root,width=root_width*0.7, height=root_height*0.4)
        self.R_frame.pack(side=tk.BOTTOM, padx=0,fill=tk.BOTH, expand=False,anchor=tk.SE)

        title_label = tk.Label(self.R_frame, text="Sensor:", font=("Arial", 16))
        title_label.pack(anchor=tk.NW)
        name_label= tk.Label(self.R_frame, text="R1    R2     R3     R4     R5    R6     R7    R8     L1    L2      L3    L4      L5    L6      L7     L8", font=("Arial", 16))
        name_label.pack(anchor=tk.NW,  pady=10) 
        # 创建实时电阻文本框
        self.real_time_R_text = tk.Text(self.R_frame,wrap=tk.WORD, font=("Arial", 16),width=72,height=12)
        self.real_time_R_text.pack(fill=tk.BOTH, expand=False)  # 填充整个右侧区域

        if self.hand=='left':
            self.wl=14  #归一化后的波长
            if self.num=='nonum':
                self.L=['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
            else:
                self.L=['1','2','3','4','5','q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
        else:
            self.wl=17  #归一化后的波长
            if self.num=='nonum':
                self.L=['y','u','h','j','n','m','i','k','l','o','p','+']
            else:
                self.L=['6','7','8','9','0','y','u','h','j','n','m','i','k','l','o','p','+']

        # self.main()
        # self.receive1data=[]

    def recv(self):
            while True:
                data =self.serial.read()#读一个字节数
                if data ==b'\x04':#如果读到第1个是04，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\xff':#如果读到第2个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x1b':#如果读到第3个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x1b':#如果读到第4个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x05':#如果读到第5个是05，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第6个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第7个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第8个是0，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x15':#如果读到第9个是15，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x17':#如果读到第10个是17，继续读
                        data =self.serial.read()#读一个字节数
                else:
                        break
                if data == b'\x00':#如果读到第11个是00，继续读
                        data =self.serial.read(19)#读剩下所有字节数
                else:
                        break        
        
                break  
            return data
        
    def receive_n_data(self): #接收n个data，并存入s_data返回(n*26)
        data1_13=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]                   
        s_data=[]    
        if self.tr==1:
            with open('data_new.csv',"r") as f0:
                data1 = list(csv.reader(f0))
            data=data1[self.nn*(self.ii-1):self.nn*self.ii]
            d=data[0]
            d=d[0:8] + d[13:21]
            d=[str(x).rjust(4, '0') for x in d]
            real_time_R_value =str(d).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
            self.real_time_R_text.insert(tk.END, "\n"+real_time_R_value)      
            self.real_time_R_text.see(tk.END)
            self.real_time_R_text.update_idletasks()
            s_data=np.array(data,dtype=float) 
            return s_data                    
        while len(s_data)<self.nn:
                d=list(range(0,26))
                data =self.recv()
                if(len(data)>15 & data[0]==0):
                        for i in range (1,len(data)):
                                data1_13[i-1]=data[i]
                else:
                        if(len(data)>15 & data[0]==1):
                                for i in range (1,9):
                                        data1_13[17+i]=data[i]
                                        
                                for i in range (11,19):
                                        data1_13[43+i-10]=data[i]   
                        else:
                                if(len(data)>15 & data[0]==2):
                                        for i in range (1,19):
                                                data1_13[26+i-1]=data[i] 

                                for i in range(0,26):
                                        d[i]=data1_13[i*2+1]*256+data1_13[i*2]
                                    
                                
                                dis=d[0:8] + d[13:21]
                                dis=[str(x).rjust(4, '0') for x in dis]
                                real_time_R_value = str(dis).replace("(", "").replace(")", "").replace("'", "").replace("[", "").replace("]", "").replace(" ", "")
                                self.real_time_R_text.insert(tk.END, "\n"+real_time_R_value)                                         
                                self.real_time_R_text.see(tk.END)                    
                                self.real_time_R_text.update_idletasks()
                                s_data.append(d)
                                # print(len(s_data),':',d)
        return s_data    
    
    def plot_conn(self):
    #获取串口列表
        port_list = list(serial.tools.list_ports.comports())
        print(port_list)
        self.Open=0
        # self.filename = 'data_new.csv' # 文件名
        
        if len(port_list)==0:
                print('无可用串口')
        else:
                for i in range(0,len(port_list)):
                        print(port_list[i])
                        if('COM5' in port_list[i]):
                                self.serial = serial.Serial('COM5', 115200, timeout=200)  #填入实际串口号
                                print("serial open success")
                                self.Open=1

    def plot_test(self):
        if (self.Open==1) :
            #
            # xis=list(range(100))
            # stat=0                 
            print('\n请进行手势：')
            s_data=[]                    
            s_data=np.array(self.receive_n_data()) #100行一个手势
            s_data=s_data.astype(np.int16)

            da1=s_data.astype(np.int16)
            davg_l=[0,0,0,0,0,0,0,0]
            davg_r=[0,0,0,0,0,0,0,0]

            for i in range(8):
                llr=da1[:,i]
                davg_r[i]=max(llr)-min(llr)
                lll=da1[:,i+13]
                davg_l[i]=max(lll)-min(lll)

            davg_l0=max(davg_l)
            davg_r0=max(davg_r)
            print('davg_l:',davg_l)
            print('davg_r:',davg_r)
            if davg_r0>davg_l0:
                  self.hand='right'
                  print('right hand')
            else:
                  self.hand='left'
                  print('left hand')

            if self.hand=='left':
                  self.h0=self.h0_l
                  self.h1=self.h1_l
                  self.wl=self.wl_l
                  self.fn2=self.fn21
                  self.max1=self.max_l
                  self.min1=self.min_l
            else:
                  self.h0=self.h0_r
                  self.h1=self.h1_r
                  self.wl=self.wl_r
                  self.fn2=self.fn22
                  self.max1=self.max_r
                  self.min1=self.min_r  


            da1 = np.hstack((da1[:, 0:8], da1[:, 13:21]))

            self.plot_sensor.clear()
            self.plot_sensor.plot(range(len(da1)), da1)
            #   # 修改为您的实际列名称
            column_labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8','L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
            self.plot_sensor.set_title("Real-time sensing data (Ω)", loc='left', pad=10)
            self.plot_sensor.set_xlabel("Data Point (Count)")
            self.plot_sensor.legend(column_labels,loc='upper left', bbox_to_anchor=(1, 1.1),fontsize='x-small')  # 添加图例，并设置位置在图像外的右上角
            self.canvas_sensor.draw()
            
            # self.root.update_idletasks()   
            self.root.update()


            if self.hand=='left':
                da1=da1[:,8:16]  
            else:
                da1=da1[:,0:8]
            # print(da1)

            #fn2,da1
            with open(self.fn4,'a+',newline="") as f4:
                writer=csv.writer(f4)
                for line in s_data:
                    writer.writerow(line)   

            ##高通滤波
            # h_data=s_data[:,:]
            # b, a = signal.butter(8, 0.5, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
            # for i in range(26):
            #     s1=s_data[:,i]
            #     print('s1:',s1)
            #     print(len(s1))
            #     h1 = signal.filtfilt(b, a, s1)  #data为要过滤的信号
            #     print('h1:',h1)
            #     print(len(h1))
            #     for j in range(len(h1)):
            #          h_data[j,i]=h1[j]

            h_data=da1[:,:].astype(np.int16)
            for i in range(8):
                s1=da1[:,i].astype(np.int16)
                h11=self.denoise(s1)
                # print('h11:',h11)
                # print('len h11:',len(h11))
                for j in range(len(h11)):
                        h_data[j,i]=h11[j]
            
            s_data=h_data
            with open(self.fn3,'a+',newline="") as f30:
                writer=csv.writer(f30)
                for line in s_data:
                    writer.writerow(line)   
            # plt.plot(xis,da1)
            # plt.savefig('test_figs/'+str(self.ii)+'.png')
            # plt.show()
            self.wave=self.cums(s_data,self.h0,self.h1) 
            print('h0,h1:',self.h0,self.h1)  
            if self.hand=='left':
                  self.h0_l=self.h0
                  self.h1_l=self.h1
                  self.max_l=self.max1
                  self.min_l=self.min1
            else:
                  self.h0_r=self.h0
                  self.h1_r=self.h1
                  self.max_r=self.max1
                  self.min_r=self.min1
            return                                                


    def sgn(self,num):
        if (num > 0):
            return 1.0
        elif (num == 0):
            return 0.0
        else:
            return -1.0

    def wavelet_noising(self,new_df):
        data = new_df
        # data = data.values.T.tolist()  # 将np.ndarray()转为列表
        data=list(data)
        w = pywt.Wavelet('sym8')
        # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波
        [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 分解波

        length1 = len(cd1)
        length0 = len(data)

        Cd1 = np.array(cd1)
        abs_cd1 = np.abs(Cd1)
        median_cd1 = np.median(abs_cd1)

        sigma = (1.0 / 0.6745) * median_cd1
        lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
        usecoeffs = []
        usecoeffs.append(ca5)  # 向列表末尾添加对象

        #软硬阈值折中的方法
        a = 0.5

        for k in range(length1):
            if (abs(cd1[k]) >= lamda):
                cd1[k] = self.sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
            else:
                cd1[k] = 0.0

        length2 = len(cd2)
        for k in range(length2):
            if (abs(cd2[k]) >= lamda):
                cd2[k] = self.sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
            else:
                cd2[k] = 0.0

        length3 = len(cd3)
        for k in range(length3):
            if (abs(cd3[k]) >= lamda):
                cd3[k] = self.sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
            else:
                cd3[k] = 0.0

        length4 = len(cd4)
        for k in range(length4):
            if (abs(cd4[k]) >= lamda):
                cd4[k] = self.sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
            else:
                cd4[k] = 0.0

        length5 = len(cd5)
        for k in range(length5):
            if (abs(cd5[k]) >= lamda):
                cd5[k] = self.sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
            else:
                cd5[k] = 0.0

        usecoeffs.append(cd5)
        usecoeffs.append(cd4)
        usecoeffs.append(cd3)
        usecoeffs.append(cd2)
        usecoeffs.append(cd1)
        recoeffs = pywt.waverec(usecoeffs, w)
        return recoeffs


    def denoise(self,data):
        data_denoising = self.wavelet_noising(data)  #调用小波去噪函数
        return data_denoising



    def cums(self,da,h0,h1):  #n1:平均数的长度 d：传感器数据列 
        # print('采集的数据:',da)
        self.wave=0
        s_index={}
        wave_index=[]
        maxd=[]
        mind=[]
        if self.hand=='left':
            mnl=500
            mxl=2500
        else:
            mnl=580
            mxl=8000
        
        for k in range(8):
            avg=[]
            davg=[]
            l0=da[:,k]
            mind.append(min(l0))
            maxd.append(max(l0))
            if min(l0)<mnl or max(l0)>mxl:
            # if minl<600:#right
                print('数据有错误')
                return -1

            for i in range(self.nn-self.lw):
                avg.append(round(sum(l0[i:i+self.lw-1])/self.lw,2))
                if i>0:
                    davg.append(round(avg[i]-avg[i-1],2))
                    if abs(avg[i]-avg[i-1])>h0:
                        s_index[i-1] = s_index.get(i-1, 0) + 1
                        wave_index.append(i-1)
        if len(wave_index)==0:
                print('波动幅度太小，找不到波')
                # print(len(s_index))
                # print(s_index)
                # print('davg:',davg)
                self.h0=max(1,self.h0-1)
                print('h0,h1:',self.h0,self.h1)
                return -4
        # if max(wave_index)-min(wave_index)>33:
        if max(wave_index)-min(wave_index)>50:
                print('wave_index:',wave_index)
                print('具有两个以上的波,h0=%d,h1=%d'%(self.h0,self.h1))
                self.h0+=1
                return -5

        endkey=[]
        for key,value in s_index.items():
            if value==max(s_index.values()):
                skey=key
                break
        n1=self.lw
        nn=self.nn
        for k in range(8):
            avg=[]
            l0=da[:,k]
            for i in range(max(0,skey-15-self.lw),min(skey+20-n1,nn-n1)):
                j=i-max(0,skey-15-n1)
                avg.append(round(sum(l0[i:i+n1-1])/n1,2))
                if j>0:
                #     davg.append(round(avg[i]-avg[i-1],2))
                    if abs(avg[j]-avg[j-1])>h1:
                    # s_index[i-1] = s_index.get(i-1, 0) + 1
                        endkey.append(i-1)  #只要8个传感器中有一个波动，就记录下来

        # print(endkey)
        if len(endkey)==0:
            self.h1=max(self.h1-1,1)
            print('找到波峰，但并未找到完整波形,skey=',skey)
            print('s_index:',s_index)
            return -3              

        s=min(endkey)+n1-1
        e=max(endkey)+n1-1
        #e=e+n1+2
        if self.hand=='left':
            aa=4
            bb=25
        else:  #right
            aa=10
            bb=45
        if aa<e-s<bb:  
            self.wave_org=da[s:e,:]
            #print(sens)
            print('找到的波长：',e-s+1,'始末位置：',s,e)
            if e-s+1<self.wl-1:
                  self.h1=self.h1-6/sqrt(self.ii)
            if e-s+1>self.wl+1:
                  self.h1=self.h1+6/sqrt(self.ii)
            if self.ii>=80:
                self.sum_wave_len+=e-s+1
                self.times+=1
            for k in range(8):
                self.max1[k]=max(self.max1[k],maxd[k])
                self.min1[k]=min(self.min1[k],mind[k])
            self.wave=1
            return 1
        else:
            # print('endkey:',endkey)
            print('s,e,e-s:',s,e,e-s)
            print('波开始和结束的位置不正确')
            if e-s<=aa:
                  self.h1=max(1,self.h1-1)
            if e-s>=bb:
                  self.h1=self.h1+1
            return -2

    def reg(self):
        da=np.array(self.wave_org)
        l1=len(da)
        de=[]
        x=[val for val in range(l1)]
        x_new=[]
        l=self.wl
        for i in range(l):         
                x_new.append(i/(l-1)*(l1-1))
        for j in range(8):
                line=[]
                line1=[]
                # sen=da[:,j].astype(np.float16)
                sen=da[:,j].astype(np.int_)
                line.append(sen[0])
                # f5 = interp1d(x, sen, kind="cubic")
                f5=CubicSpline(x,sen)
                line1=np.round(f5(x_new),2)
                de.append(line1)
   
        de=np.array(de)  #插值后的data转置
        self.cubic=de.T  #长度归一化后的data

    def regular_whole(self):
        data=np.array(self.cubic,dtype=float)
        max1=self.max1
        min1=self.min1
        print('max:',max1)
        print('min:',min1)
        n=len(data)
        d=[]
        for i in range(n):
            line=[]
            for j in range(8):
                line.append((data[i,j]-min1[j])/(max1[j]-min1[j]))
            d.append(line)
        self.regularized=np.array(d)  #整体归一化后的测试数据

    def svm_test(self):
        # if self.hand=='left':
        #     len_data=14
        # else:
        #     len_data=17
        len_data=self.wl
        hand=self.hand
        num=self.num
        lda_path = os.path.join('pretrained_model', f"lda_{hand}_{num}.model")
        model_path = os.path.join('pretrained_model', f"model_{hand}_{num}.model")
        svm_path = os.path.join('pretrained_model', f"svm_{hand}_{num}.model")
        if hand=='right':
            if num=='nonum':
                c_err=[('m','n'),('y','u','h'),('y','u'),('h','j')] 
            else:
                c_err=[('o','9'),('0','p'),('m','n'),('y','u','h'),('y','u'),('h','j'),('y','6'),('8','i')]
        else:
            if num=='nonum':
                c_err=[('r','t'),('t','g')]
            else:
                c_err=[] 

        # 加载模型
        with open(lda_path, 'rb') as f:
            lda_loaded = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)

        data = self.regularized
        X=[]
        for i in range(0,1):
            x=[]
            for j in range(0,8):#8列数据
                height=data[len_data*i:len_data*i+len_data,j].tolist()
                x.extend(height)
            X.append(x)

        xtest=lda_loaded.transform(X)
        output=model.predict(xtest)

        for k in range(len(c_err)):
            if output in c_err[k]:
                output=(svm_model[c_err[k]].predict(X))[0]
 
        output = str(output).replace("[", "").replace("]", "").replace("'", "").replace("+", " ")
        text_length = len(self.final_predict_text.get("1.0", "end-1c"))
        if text_length >= 26:
              self.final_predict_text.delete("1.0", "end")
        self.final_predict_text.insert(tk.END, output)
        self.final_predict_text.tag_configure("tag", font=("Arial", 60))
        self.final_predict_text.tag_add("tag", "1.0", tk.END)
        self.final_predict_text.see(tk.END)
        self.final_predict_text.update_idletasks()
        # self.root.update_idletasks()   
        # self.root.update()
        # print(output)
        # pp = pyttsx3.init()
        # pp.say(output)
        # pp.runAndWait()
        if output != " ":
            self.out+=output
        else:
            if self.out:
                pp = pyttsx3.init()
                pp.say(self.out)
                pp.runAndWait()
                self.out= ""


    def exit_app(self, event=None):
        self.root.quit()  # 退出应用
        self.root.destroy()  # 销毁主窗口

    def main(self):
        if os.path.exists(Md.fn4):
            os.remove(Md.fn4)
        if os.path.exists(Md.fn3):
            os.remove(Md.fn3)
        if os.path.exists(Md.fn21):
            os.remove(Md.fn21)
        if os.path.exists(Md.fn22):
            os.remove(Md.fn22)

        while(1):               
            if self.run==0:
                break
            self.plot_test()  # 数据输入
            if self.wave == 1:
                self.reg()  # cubic
                self.regular_whole()
                self.svm_test()

            self.ii += 1  
        if self.ii>80:
            print('从第80次开始，找到的波的平均波长：',self.sum_wave_len/self.times)




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Md = glv()

    if Md.tr == 0:
        Md.plot_conn()  # 这部分完成手套的检验
    else:
        Md.Open = 1

    # 启动主程序
    Md.main()
    # if Md.ii>80:
    #     print('从第80次开始，找到的波的平均波长：',Md.sum_wave_len/Md.times)
