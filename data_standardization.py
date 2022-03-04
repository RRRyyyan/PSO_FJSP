# coding: utf-8

import random
import sys
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from math import sqrt


sys.path.append("D:\排程程式碼\PSO_scheduling") #將資料夾的路徑放入sys.path

# 自己寫的file
from FitnessFunction import move_fitness1, Schedule, Utilize_rate, Load_balance, On_time_fit

#讀取mk01檔案並將字串轉成整數
def get_data(path):
    f = open(path,"r")
    mk01_data = []
    for line in f.readlines():
        line=line.strip('\n') #去除尾部的換行符
        mk01_data.append(line)
    f.close()


    for i in range(len(mk01_data)):
        mk01_data[i] = mk01_data[i].split() #以空格分割字串成list
        for j in range(len(mk01_data[i])):
            mk01_data[i][j] = int(mk01_data[i][j]) #將str轉int
    return mk01_data


#定義每個JOB的製程數
# Job_Op_Num = np.array([6,5,5,5,6,6,5,5,6,6])
Job_Op_Num = np.array([6, 6, 6, 6, 6, 6, 5, 6, 5, 6])

#定義客戶交期(預設)
customer_date = np.array([40 for i in range(10)])

#定義PSO演算法函式
class PSOSolver():
    def __init__(self,particle_number,machine, job, vmax, iterations,
               cognition_factor,social_factor, inertial_factor):
        
        #####需自行設定之參數#####
        self.particle_number = particle_number #粒子數 
        self.machine = machine #機台數
        self.job = job #工單數
        # self.cognition_factor = np.linspace(cognition_factor[0],cognition_factor[1],iterations)
        # self.social_factor = np.linspace(social_factor[0],social_factor[1],iterations)
        self.cognition_factor = cognition_factor #自身權重
        self.social_factor = social_factor  #群體權重
        self.inertial_factor = inertial_factor #慣性權重
        self.iterations = iterations #迭代次數
        self.fitness_part = [] # 適應度
        self.move_fit = []
        self.inv_utilize_fit = []
        self.load_balance_std = []
        self.on_time_fit = []
        
        
        #####定義粒子屬性#####
        self.solutions = [[[] for j in range(self.job)] for i in range(self.particle_number)] #當前位置
        self.pbest = [] #局部最佳位置
        self.pbest_obj = [] #局部最佳適應值
        
        self.gbest = [] #全局最佳位置
        self.gbest_obj = 0.0;
    
        self.current_v = [] #粒子當前方向
                
        
        
        
        self.count = [[0 for i in range(self.machine)] for j in range(self.particle_number)] #計算各機台處理的OP數
        self.op_machine_pair = [[[] for j in range(self.job)]for i in range(self.particle_number)] #每道OP對應到的機台編號
        self.each_machine_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #記錄機台OP時間序
        self.record_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #紀錄{Job,op,st,et}
        self.part_val = [[] for i in range(self.particle_number)] #存放每次迭代各粒子的適應度
        self.OPS = [[0 for j in range(np.sum(Job_Op_Num[:self.job]))]for i in range(self.particle_number)] #存放各粒子製程的操作順序
        self.pbest_OPS = [[] for i in range(self.particle_number)] #局部最佳的製程順序
        self.job_date = [np.zeros(self.job) for i in range(self.particle_number)]
        self.max_seq_index = [] #紀錄最晚結束時間的[machine, ind]

        
    #####初始化#####    
    def Initialize(self):
        
        self.solutions = [[[] for j in range(self.job)] for i in range(self.particle_number)] #當前位置
        self.op_machine_pair = [[[] for j in range(self.job)]for i in range(self.particle_number)] #每道OP對應到的機台編號
        self.each_machine_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #記錄機台OP時間序
        
        
        
        
        min_index = 0 #最小適應度的粒子編號
        min_val = sys.float_info.max #將初始最小適應值設為最大浮點數值
        
        #####初始化位置#####
        for i in range(self.particle_number):
#             print(f"Partical{i+1}:")
            total_op = 0 #用於讀取mk01 data的累加數
            for j in range(self.job):
#                 print(f"JOB{j+1}:{Job_Op_Num[j]}道OP")
                input_data = np.array([[0 for _ in range(self.machine)] for _ in range(Job_Op_Num[j])],dtype='float64')
                for k in range(Job_Op_Num[j]):
                    rnd = random.randint(1,mk01_data[total_op+k][0]) #隨機選擇該製程可用機台數量編號之一
#                     print(f"The No.{k+1} operation choose machine {mk01_data[total_op+k][rnd*2-1]}, cost time:{mk01_data[total_op+k][rnd*2]}, ")
                    input_data[k][mk01_data[total_op+k][rnd*2-1]]=1
#                     print(f"The input data:{input_data[k]}\n")
                
                self.OPS[i][total_op:total_op+Job_Op_Num[j]]=[j for i in range(Job_Op_Num[j])]
                self.solutions[i][j].append(input_data) #solutions資料維度為:partical_number*Job*該Job製程數*機台數
                total_op+=Job_Op_Num[j] #累加OP的數量
            
            self.solutions[i] = np.squeeze(self.solutions[i]) #去除為1的維度
            self.OPS[i] = np.random.choice(self.OPS[i],size=len(self.OPS[i]),replace=False) #打亂工單順序
            
            
            

        
        ######計算每個OP分配到各機台的總數&製程排列順序#####
        self.Cal_count_opair_opseq()
        

    
        #####更新局部最佳適應度為初始適應度#####
        for i in range(self.particle_number):
            
            # obj_val, job_op_seq = schedule(self.op_machine_pair[i], self.each_machine_time[i], self.OPS[i], self.record_time[i])
            fitness, job_op_seq = self.Fitness_func(i, self.OPS[i], self.op_machine_pair[i] )
           
                
    #####計算某特定[job,op]的花費時間#####
    def Find_op_time(self,array,m):
        ct = 0
        for k in range(1,len(mk01_data[array[1]+sum(Job_Op_Num[:array[0]])]),2):
                    if (mk01_data[array[1]+sum(Job_Op_Num[:array[0]])][k]==m):
                        ct = mk01_data[array[1]+sum(Job_Op_Num[:array[0]])][k+1]
        if ct==0:
            ct=999
        return ct           
        

    
    
    #計算每個OP分配到各機台的總數&製程機台配對&各機台處理的製程
    def Cal_count_opair_opseq(self):
        for i in range(self.particle_number):
            for j in range(self.job):
                for k in range(Job_Op_Num[j]):
                    index = np.argmax(self.solutions[i][j][k])
                    self.count[i][index]+=1
                    self.op_machine_pair[i][j].append(index)
    
    
    
                
    #####統整fitness#####
    def Fitness_func(self, indx, OPS, machine_assign,solution, lbs_weight=0.1, iu_weight=0.2, mo_weight=0.3, otf_weight=0.4):
        # 考慮不同指標的適應度
        move_fit = move_fitness1(self.solutions[indx]) # 搬運成本
        max_end_time, job_op_seq, job_finish = Schedule(machine_assign, self.each_machine_time[indx], OPS, self.record_time[indx], self.machine, self.job, self.Find_op_time, Job_Op_Num) #結束時間
        inv_utilize_fit = Utilize_rate(self.each_machine_time[indx], max_end_time, self.machine)*100 # 反稼動率
        load_balance_std = Load_balance(self.each_machine_time[indx], self.machine)
        on_time_fit = On_time_fit(customer_date, job_finish, self.job)
        
        # 各個粒子適應度開根號減少之間scale的差距
        total_fitness = (mo_weight*move_fit+ 
                          iu_weight*inv_utilize_fit+
                          lbs_weight*load_balance_std+
                          otf_weight*on_time_fit)
        
        # # 各個粒子適應度開根號減少之間scale的差距
        # total_fitness = max_end_time
        
        self.move_fit.append(move_fit)
        self.inv_utilize_fit.append(inv_utilize_fit)
        self.load_balance_std.append(load_balance_std)
        self.on_time_fit.append(on_time_fit)
        
        
        self.fitness_part.append([move_fit,inv_utilize_fit,load_balance_std, on_time_fit] )
        
        return total_fitness, job_op_seq



######################################################################################################################
######################################
###############開始使用################
######################################
######################################################################################################################
mk01_data = get_data("MK02.txt")
t = 0
solver = PSOSolver(particle_number=100,machine=6, job=10,vmax=0.5, iterations = 2, cognition_factor=2, social_factor=2, inertial_factor=0.8)
with open('data_standard.txt', 'w') as f:
    data_standard = {}
    for j in range(100):
        solver.move_fit = []
        solver.inv_utilize_fit = []
        solver.load_balance_std = []
        solver.on_time_fit = []
        mf_m = []
        iuf_m = []
        lbs_m = []
        otf_m = []
        mf_d = []
        iuf_d = []
        lbs_d = []
        otf_d = []
        
        print('第{}次'.format(j+1))
        solver.Initialize()
        mf_avg = sum(solver.move_fit)/len(solver.move_fit)
        iuf_avg = sum(solver.inv_utilize_fit)/len(solver.inv_utilize_fit)
        lbs_avg = sum(solver.load_balance_std)/len(solver.load_balance_std)
        otf_avg = sum(solver.on_time_fit)/len(solver.on_time_fit)
        
        mf_m.append(mf_avg)
        iuf_m.append(iuf_avg)
        lbs_m.append(lbs_avg)
        otf_m.append(otf_avg)
        
        mf_d.append(np.std(solver.move_fit))
        iuf_d.append(np.std(solver.inv_utilize_fit))
        lbs_d.append(np.std(solver.load_balance_std))
        otf_d.append(np.std(solver.on_time_fit))
        
        # data_standard['move_fit'] = (round(min(solver.move_fit),2), round(max(solver.move_fit),2))
        # data_standard['inv_utilize_fit'] = (round(min(solver.inv_utilize_fit),2), round(max(solver.inv_utilize_fit),2))
        # data_standard['load_balance_std'] = (round(min(solver.load_balance_std),2), round(max(solver.load_balance_std),2))
        # data_standard['on_time_fit'] = (round(min(solver.on_time_fit),2), round(max(solver.on_time_fit),2))
        
    mf_final_avg = sum(mf_m)/len(mf_m)
    iuf_final_avg = sum(iuf_m)/len(iuf_m)
    lbs_final_avg = sum(lbs_m)/len(lbs_m)
    otf_final_avg = sum(otf_m)/len(otf_m)
    
    mf_final_d = sum(mf_d)/len(mf_d)
    iuf_final_d = sum(iuf_d)/len(iuf_d)
    lbs_final_d = sum(lbs_d)/len(lbs_d)
    otf_final_d = sum(otf_d)/len(otf_d)
    
    data_standard['move_fit'] = (round(mf_final_avg,2), round(mf_final_d,2))
    data_standard['inv_utilize_fit'] = (round(iuf_final_avg,2),round(iuf_final_d,2))
    data_standard['load_balance_std'] = (round(lbs_final_avg,2),round(lbs_final_d,2))
    data_standard['on_time_fit'] = (round(otf_final_avg,2),round(otf_final_d,2))
    
    f.write(str(data_standard))
