#!/usr/bin/env python
# coding: utf-8


import random
import sys
import numpy as np
import time
import copy
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) #去除warning
sys.path.append("D:\排程程式碼\PSO_scheduling")  # 將資料夾的路徑放入sys.path
from FitnessFunction import move_fitness1, Schedule, Utilize_rate, Load_balance, On_time_fit, move_fit_nor, utilize_fit_nor, Load_balance_nor, On_time_fit_nor



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
Job_Op_Num = np.array([6, 5, 5, 5, 6, 6, 5, 5, 6, 6])

onematrix = np.ones(10, dtype=int)
#定義客戶交期(預設)
customer_date = onematrix*42
# customer_date = np.array([38, 38, 38, 38, 38, 38, 38, 38, 38, 38])


class PSOSolver():
    def __init__(self,particle_number,machine, job, iterations,
               cognition_factor,social_factor, inertial_factor,data, Job_Op_Num, request_date):
        
        #####需自行設定之參數#####
        self.particle_number = particle_number #粒子數 
        self.machine = machine #機台數
        self.job = job #工單數
        # self.cognition_factor = np.linspace(cognition_factor[0],cognition_factor[1],iterations)
        # self.social_factor = np.linspace(social_factor[1],social_factor[0],iterations)
        self.cognition_factor = cognition_factor #自身權重
        self.social_factor = social_factor  #群體權重
        self.inertial_factor = inertial_factor #慣性權重
        self.iterations = iterations #迭代次數
        
        #####定義粒子屬性#####
        self.solutions = [[[] for j in range(self.job)] for i in range(self.particle_number)] #當前位置
        self.pbest = [] #局部最佳位置
        self.pbest_obj = [] #局部最佳適應值
        self.gbest = [] #全局最佳位置
        self.gbest_obj = 0.0
    
        self.current_v = [] #粒子當前方向
        self.data = data
        self.Job_Op_Num = Job_Op_Num
        self.request_date = request_date
                
        
        
        
        self.count = [[0 for i in range(self.machine)] for j in range(self.particle_number)] #計算各機台處理的OP數
        self.op_machine_pair = [[[] for j in range(self.job)]for i in range(self.particle_number)] #每道OP對應到的機台編號
        self.each_machine_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #記錄機台OP時間序
        self.record_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #紀錄{Job,op,st,et}
        self.part_val = [[] for i in range(self.particle_number)] #紀錄每次迭代每個粒子的適應度
        self.OPS = [[0 for j in range(sum(self.Job_Op_Num[:self.job]))]for i in range(self.particle_number)]#存放每個machine的操作順序
        self.pbest_OPS = [[] for i in range(self.particle_number)] #局部最佳的machine操作順序
        self.max_seq_index = [] #紀錄最晚結束時間的[machine, ind]
        
    #####初始化#####    
    def Initialize(self):
        
        min_index = 0 #最小適應度的粒子編號
        min_val = sys.float_info.max #將初始最小適應值設為最大浮點數值
        
        #####初始化位置#####
        for i in range(self.particle_number):
#             print(f"Partical{i+1}:")
            total_op = 0 #用於讀取mk01 data的累加數
            for j in range(self.job):
#                 print(f"JOB{j+1}:{Job_Op_Num[j]}道OP")
                self.input_data = np.array([[0 for _ in range(self.machine)] for _ in range(self.Job_Op_Num[j])])
                for k in range(self.Job_Op_Num[j]):
                    rnd = random.randint(1,int(self.data[total_op+k][0])) #隨機選擇該製程可用機台數量編號之一
#                     print(f"The No.{k+1} operation choose machine {mk01_data[total_op+k][rnd*2-1]}, cost time:{mk01_data[total_op+k][rnd*2]}, ")
                    self.input_data[k][int(self.data[total_op+k][rnd*2-1])]=1
#                     print(f"The input data:{input_data[k]}\n")
                
                self.OPS[i][total_op:total_op+self.Job_Op_Num[j]]=[j for m in range(self.Job_Op_Num[j])]
                # print(f"OPS={self.OPS[i]}")
                self.solutions[i][j].append(self.input_data) #solutions資料維度為:partical_number*Job*該Job製程數*機台數
                # print(f"solutions={self.solutions[i][j]}")
                total_op+=self.Job_Op_Num[j] #累加OP的數量
            
            self.solutions[i] = np.squeeze(self.solutions[i]) #去除為1的維度(job*1*job_OP數 => job*job_OP數)
            self.OPS[i] = np.random.choice(self.OPS[i],size=len(self.OPS[i]),replace=False) #打亂工單順序

        
        #計算每個OP分配到各機台的總數(self.count&self.OP_machine_pair)
        self.Cal_count_opair_opseq()
        
        
        
        
        
        



        #####更新局部最佳適應度為初始適應度#####
        for i in range(self.particle_number):
            
            print("\n")
            fitness, job_op_seq,fa = self.Fitness_func(i, self.OPS[i], self.op_machine_pair[i], self.solutions[i])
            print("=================================================")
            print(f"The totoal time of particle {i+1} = {fitness}")
            print("=================================================\n")
            self.pbest_obj.append(fitness) #維度為:particle_number*1

            #紀錄最小適應度的粒子編號
            if(fitness < min_val):
                min_index = i
                min_val = fitness
                self.best_job_op_seq = copy.deepcopy(job_op_seq)
            self.pbest_OPS[i] = copy.deepcopy(self.OPS[i])
        
        #####更新局部最佳機台配對為初始化機台配對#####
        self.pbest_machine_assign = copy.deepcopy(self.op_machine_pair)

        
        #####更新局部最佳位置為初始化位置#####
        self.pbest = copy.deepcopy(self.solutions) #用深複製避免一次修改兩個資料
        

        
        #####初始化方向速度設為與初始位置相同#####
        self.current_v = copy.deepcopy(self.solutions)
        

        
        #####更新全局最佳適應值以及最佳位置#####
        
        self.gbest_obj = min_val
        self.gbest = copy.deepcopy(self.solutions[min_index])
        self.gbest_OPS = copy.deepcopy(self.OPS[min_index])
        self.gbest_OPS_for_POX = copy.deepcopy(self.gbest_OPS)
        self.gbest_time_seq = copy.deepcopy(self.each_machine_time[min_index])
        self.gbest_machine_assign = copy.deepcopy(self.op_machine_pair[min_index])
        self.best_arange = copy.deepcopy(self.record_time[min_index])
        self.gbest_solution_POX = copy.deepcopy(self.solutions[min_index])
        ###############################################################################################
                
                
        print("初始化粒子完成")
#         print(f"\n初始全局最佳位置:\n{self.gbest}")
        print(f"\n初始全局最佳OPS:\n{self.gbest_OPS}")
        print(f"初始全局最佳適應度:{self.gbest_obj}")
        print("開始迭代!\n")
        
        
    #####更新粒子位置#####    
    def Move_to_new_positions(self, iter_num):
        for i in range (self.particle_number):
            alpha = self.cognition_factor*random.random() #C1*rand()
            beta = self.social_factor*random.random() #C2*rand()
                
            #####更新粒子方向#####
            ###加入慣性權重###
            self.current_v[i] = self.current_v[i]+alpha*(self.pbest[i]-self.solutions[i])+beta*(self.gbest-self.solutions[i])


            # #####丟入Sigmoid函數更新位置(輸出0或1)#####  
            # for j in range(self.job):
            #     rand_array = np.array([[random.random() for i in range(self.machine)]for j in range(len(self.current_v[i][j]))])
            #     self.solutions[i][j][self.Sigmoid(self.current_v[i][j])>rand_array]=1  #與Sigmoid函示比較，輸出0或1
            #     self.solutions[i][j][self.Sigmoid(self.current_v[i][j])<rand_array]=0

            #     for k in range(len(self.solutions[i][j])):
            #         if(np.count_nonzero(self.solutions[i][j][k]==1))==0: #若輸出都完全沒有1
            #             s = random.randrange(0,self.machine) #隨機指派一個機器為1
            #             self.solutions[i][j][k][s] = 1
            #         elif(np.count_nonzero(self.solutions[i][j][k]==1))>1: #若有複數個1
            #             idx = np.where(self.solutions[i][j][k]==1)[0] #找出有哪些位置為1
            #             self.solutions[i][j][k][self.solutions[i][j][k]==1]=0 #將1的位置全設為0
            #             s = random.choice(idx) #從1的編號中隨機選一個機台為1
            #             self.solutions[i][j][k][s] = 1
            
            
            #####丟入sigmoid函數更新位置(輸出0或1)#####  
            for j in range(self.job):
                X = np.zeros(np.shape(self.solutions[i][j]))
                for k in range(len(self.solutions[i][j])):
                    
                    ma_num = int(self.data[sum(self.Job_Op_Num[:j])+k][0])
                    ma_can_use = []
                    for s in range(1,ma_num+1,1):
                        ma_can_use.append(int(self.data[sum(self.Job_Op_Num[:j])+k][s*2-1]))
                    order = np.random.choice(ma_can_use, size=len(ma_can_use),replace=False) #打亂機台編號
                    
                    for l in range(len(order)):
                        r = random.random()
                        tmp = self.Sigmoid(self.current_v[i][j][k][order[l]])
                        if r<tmp:
                            X[k][order[l]]=1 #只要有一個位置為1，就跳出迴圈
                            break
                        elif l==(len(order)-1): #若都沒有1，則將最後一個機台設為1
                            X[k][order[l]]=1
                            
                self.solutions[i][j]=copy.deepcopy(X)
            
           
    
    #將每個particle的OPS與gbest_OPS做POX & 每隔10次迴圈就檢查critical path並挑選其中的OP重做machine assignment            
    def Swap_op_seq(self):
        for i in range(self.particle_number):
            for k in range(0,30,5):
                
                #每跑10次迴圈就交換一次gbest的machine assignment
                if (k!=0 and k%5==0):
                    self.max_seq_index = self.Find_max_index(self.gbest_time_seq, self.gbest_obj)
                    self.critical_path = self.Find_CP(self.max_seq_index, self.best_job_op_seq, self.gbest_time_seq, self.gbest_machine_assign)
                    
                    CP_maas = [] #存放critical path交換過的machine assignment
                    CP_solution = [] #存放critical path交換過後的的solution
                    
                    for item in self.critical_path:
                        self.gbest_solution_POX = copy.deepcopy(self.gbest)
                        self.gbest_maas_for_POX = copy.deepcopy(self.gbest_machine_assign)
#                         num = random.randint(1,len(item)) #要做machine reassign的OP數目
                        To_be_change = random.sample(item,1) #隨機選擇1個製程進行交換
                        for j in range(1):
                            input_data = np.array([0 for _ in range(self.machine)])
                            op_pick = To_be_change[j] #op_pick為單一製程
                            mach = self.gbest_machine_assign[op_pick[0]][op_pick[1]] #原本選的機台
                            choose_num = int(self.data[sum(self.Job_Op_Num[:op_pick[0]])+op_pick[1]][0]) #可選的機台數
                            if choose_num!=1:
                                #若新選擇的機台與原本相同就繼續選
                                while(mach==self.gbest_machine_assign[op_pick[0]][op_pick[1]]):
                                    rnd = random.randint(1,choose_num)                               
                                    mach = int(self.data[sum(self.Job_Op_Num[:op_pick[0]])+op_pick[1]][rnd*2-1])
                            input_data[mach]=1
                            self.gbest_solution_POX[op_pick[0]][op_pick[1]] = input_data
                            #去除為1的維度
                            self.gbest_solution_POX[op_pick[0]][op_pick[1]] = np.squeeze(self.gbest_solution_POX[op_pick[0]][op_pick[1]]) #去除為1的維度
                            index = np.argmax(self.gbest_solution_POX[op_pick[0]][op_pick[1]]) 
                            self.gbest_maas_for_POX[op_pick[0]][op_pick[1]] = index
                        CP_maas.append(self.gbest_maas_for_POX)
                        CP_solution.append(self.gbest_solution_POX)
                    
                    # machine reassign後與未做POX前的gbest_OPS算fitness
                    for m in range(len(CP_maas)):
                        flag1 = self.Check_POX(i, self.gbest_OPS, CP_maas[m], CP_solution[m])
                        flag2 = self.Check_POX(i, self.OPS[i], CP_maas[m], CP_solution[m])

                    
                    for f in range(5):
                        self.gbest_OPS_for_POX = copy.deepcopy(self.gbest_OPS)
                        origin_OPS = copy.deepcopy(self.OPS[i])
                        s1_index = [] #存放OPS中被SUB2挑選的Job編號位置
                        s2_index = [] 
                        w = np.arange(0,self.job)
                        num = random.randint(1,self.job-2) #有幾個job要固定(1~8個)=>變動的job數為2~9個
                        x = np.random.choice(w,size=num,replace=False)  #求SUB1的Job編號
                        for j in range(len(self.OPS[0])):
                            if(self.gbest_OPS_for_POX[j] not in x):
                                s1_index.append(j)
                            if(self.OPS[i][j] not in x): 
                                s2_index.append(j)
                        
                        # gbest與每一particle的OPS做交換
                        for n in range(len(s1_index)):
                            self.gbest_OPS_for_POX[s1_index[n]], self.OPS[i][s2_index[n]]= self.OPS[i][s2_index[n]], self.gbest_OPS_for_POX[s1_index[n]] 
                        
                        #若rand<0.01，進行突變
                        if (random.random()<0.01):
                            rnd = random.randint(0,45)
                            temp = self.OPS[i][rnd:rnd+10]
                            random.shuffle(temp)
                            self.OPS[i][rnd:rnd+10] = temp
                        if (random.random()<0.01):
                            rnd = random.randint(0,45)
                            temp = self.gbest_OPS_for_POX[rnd:rnd+10]
                            random.shuffle(temp)
                            self.gbest_OPS_for_POX[rnd:rnd+10] = temp
                        
                        # 將找到的每條critical path的machine assignment與交換過的OPS計算fitness
                        # 若該OPS未更新到pbest/gbest，就不替換掉(flag2=False)
                        for m in range(len(CP_maas)):
                            flag1 = self.Check_POX(i, self.gbest_OPS_for_POX, CP_maas[m], CP_solution[m])
                            flag2 = self.Check_POX(i, self.OPS[i], CP_maas[m], CP_solution[m])
                            if flag2==False:
                                self.OPS[i] = origin_OPS
                
                else:
                    for f in range(5):
                        self.gbest_solution_POX = copy.deepcopy(self.gbest)
                        self.gbest_OPS_for_POX = copy.deepcopy(self.gbest_OPS)
                        self.gbest_maas_for_POX = copy.deepcopy(self.gbest_machine_assign)
                        origin_OPS = copy.deepcopy(self.OPS[i])
                        s1_index = [] #存放OPS中被SUB2挑選的Job編號位置
                        s2_index = [] 
                        w = np.arange(0,self.job)
                        num = random.randint(1,self.job-2) #有幾個job要固定(1~8個)=>變動的job數為2~9個
                        x = np.random.choice(w,size=num,replace=False)  #求SUB1的Job編號
                        for j in range(len(self.OPS[0])):
                            if(self.gbest_OPS_for_POX[j] not in x):
                                s1_index.append(j)
                            if(self.OPS[i][j] not in x): 
                                s2_index.append(j)

                        for n in range(len(s1_index)):
                            self.gbest_OPS_for_POX[s1_index[n]], self.OPS[i][s2_index[n]] = self.OPS[i][s2_index[n]], self.gbest_OPS_for_POX[s1_index[n]] 
                            
                        #若rand<0.01，進行突變
                        if (random.random()<0.01):
                            rnd = random.randint(0,45)
                            temp = self.OPS[i][rnd:rnd+10]
                            random.shuffle(temp)
                            self.OPS[i][rnd:rnd+10] = temp
                            
                        if (random.random()<0.01):
                            rnd = random.randint(0,45)
                            temp = self.gbest_OPS_for_POX[rnd:rnd+10]
                            random.shuffle(temp)
                            self.gbest_OPS_for_POX[rnd:rnd+10] = temp
                            
                        flag1 = self.Check_POX(i, self.gbest_OPS_for_POX, self.gbest_maas_for_POX, self.gbest_solution_POX)
                        flag2 = self.Check_POX(i, self.OPS[i], self.gbest_maas_for_POX, self.gbest_solution_POX)
                        if flag2==False:
                            self.OPS[i] = origin_OPS
    
    #計算做完POX後的fitness
    def Check_POX(self, ind ,OPS, machine_assign, solution):
        
        #重新計算以下參數
        self.each_machine_time[ind] = [[] for j in range(self.machine)]#排時間
        self.record_time[ind] = [[] for j in range(self.machine)]
        flag = False
        fitness, job_op_seq,fa = self.Fitness_func(ind, OPS, machine_assign,solution)
        
        #更新局部粒子最佳位置
        if(fitness < self.pbest_obj[ind]):
            self.pbest_obj[ind] = fitness
            self.pbest[ind] = copy.deepcopy(solution)
            if ((OPS==self.OPS[ind]).all()):
                self.pbest_OPS[ind] = copy.deepcopy(OPS)
                flag = True
            self.pbest_machine_assign[ind] = copy.deepcopy(machine_assign)

        #更新全局最佳位置
        if(fitness < self.gbest_obj):
            self.gbest_obj = fitness
            self.gbest = copy.deepcopy(solution)
            self.gbest_OPS = copy.deepcopy(OPS)
            self.gbest_time_seq = copy.deepcopy(self.each_machine_time[ind])
            self.gbest_machine_assign = copy.deepcopy(machine_assign)
            self.best_arange = copy.deepcopy(self.record_time[ind])
            self.best_job_op_seq = copy.deepcopy(job_op_seq)
            flag = True
            print(f"Find better fitness {self.gbest_obj}")
            print(fa)
        return flag

    
    #####更新最佳適應度與位置#####
    def Update_best_solution(self):
        # print("==================================")
        # print("START RENEW SOLUTION")
        # print("==================================")

        #每次疊代都要重新計算以下參數
        self.count = [[0 for i in range(self.machine)] for j in range(self.particle_number)] #計算各機台處理的OP數
        self.op_machine_pair = [[[] for j in range(self.job)]for i in range(self.particle_number)] #每道OP對應到的機台編號
        self.each_machine_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #排時間
        self.record_time = [[[] for j in range(self.machine)] for i in range(self.particle_number)] #紀錄{Job,op,st,et}
        
        self.Cal_count_opair_opseq() #重新計算op_machine_pair

        
        for i,location in enumerate(self.solutions):
            fitness, job_op_seq, fa = self.Fitness_func(i, self.OPS[i], self.op_machine_pair[i], location)
            self.part_val[i] = fitness
            #更新局部粒子最佳位置
            if(fitness < self.pbest_obj[i]):
                self.pbest_obj[i] = fitness
                self.pbest[i] = copy.deepcopy(location)
                self.pbest_OPS[i] = copy.deepcopy(self.OPS[i])
                self.pbest_machine_assign[i] = self.op_machine_pair[i]
            
            #更新全局最佳位置
            if(fitness < self.gbest_obj):
                self.gbest_obj = fitness
                self.gbest = copy.deepcopy(location)
                self.gbest_OPS = copy.deepcopy(self.OPS[i])
                self.gbest_time_seq = copy.deepcopy(self.each_machine_time[i])
                self.gbest_machine_assign = copy.deepcopy(self.op_machine_pair[i])
                self.best_arange = copy.deepcopy(self.record_time[i])
                self.best_job_op_seq = copy.deepcopy(job_op_seq)
                print(f"Find better fitness {self.gbest_obj}")
                print(fa)




    
    
    #計算某特定[job,op]的花費時間
    def Find_op_time(self,array,m):
        ct = 0
        for k in range(1,len(self.data[array[1]+sum(self.Job_Op_Num[:array[0]])]),2):
                    if (int(self.data[array[1]+sum(self.Job_Op_Num[:array[0]])][k])==m):
                        ct = self.data[array[1]+sum(self.Job_Op_Num[:array[0]])][k+1]
        if ct==0:
            ct=999
        return ct           
        

    
    
    #找最晚完成製程的機台順序編號
    def Find_max_index(self, each_machine_time, max_time):
        max_time_index = []
        for i in range(self.machine):
            if (each_machine_time[i]!=[]):
                if (each_machine_time[i][-1][1]==max_time):
                    max_time_index.append([i,len(each_machine_time[i])-1])
        return max_time_index
                    
        
    #計算每個OP分配到各機台的總數&製程機台配對
    def Cal_count_opair_opseq(self):
        for i in range(self.particle_number):
            for j in range(self.job):
                for k in range(self.Job_Op_Num[j]):
                    index = np.argmax(self.solutions[i][j][k])
                    self.count[i][index]+=1
                    self.op_machine_pair[i][j].append(index)

    
    #找critical path
    def Find_CP(self, max_seq_index, best_job_op_seq, best_time_seq, pair):
        critical_path = []
        for item in max_seq_index: #item為最大結束時間的位置index
            path = [] #存放該路徑上的[job, OP]編號
            job_op_self = best_job_op_seq[item[0]][item[1]] #最晚結束時間的job
            path.insert(0,job_op_self)  #將最晚結束時間的OP加入path
            latest_op_now = job_op_self  #紀錄當前最晚結束的[job, OP]編號
            
            #當該OP不為該機台上第一個或是不為該job第一個OP，就持續跑回圈尋找
            while(item[1]!=0 or latest_op_now[1]!=0):
                # 當該OP為該機台上第一個OP
                if (item[1]==0):
                    front_m_time=0
                    front_job_op = [latest_op_now[0], latest_op_now[1]-1] #同Job的前一個OP
                    front_job_m = pair[front_job_op[0]][front_job_op[1]] #找到其運作機台
                    front_ind = best_job_op_seq[front_job_m].index(front_job_op) #第幾順序
                    front_op_time = best_time_seq[front_job_m][front_ind][1] #結束時間
                # 當該OP為該job第一個OP
                elif (latest_op_now[1]==0):
                    front_op_time = 0
                    front_m_op = best_job_op_seq[item[0]][item[1]-1] #同machine前一個OP
                    front_m_time = best_time_seq[item[0]][item[1]-1][1] #同machine前一個OP結束時間
                else:
                    front_m_op = best_job_op_seq[item[0]][item[1]-1] #同machine前一個OP
                    front_m_time = best_time_seq[item[0]][item[1]-1][1] #同machine前一個OP結束時間
                
                    front_job_op = [latest_op_now[0], latest_op_now[1]-1] #同Job的前一個OP
                    front_job_m = pair[front_job_op[0]][front_job_op[1]] #找到其運作機台
                    front_ind = best_job_op_seq[front_job_m].index(front_job_op) #第幾順序
                    front_op_time = best_time_seq[front_job_m][front_ind][1] #同Job的前一個OP結束時間
                
                # 比較較晚的結束時間
                # 同機台前一個OP結束時間較晚
                if(front_m_time>front_op_time):
                    path.insert(0,front_m_op)
                    item = [item[0], item[1]-1]
                    latest_op_now = front_m_op
                # 同Job前一個OP結束時間較晚
                elif(front_m_time<front_op_time):
                    path.insert(0,front_job_op)
                    item = [front_job_m,front_ind]
                    latest_op_now = front_job_op
                else:# 若為相同時間，則隨機擇一
                    rnd = random.random()
                    if(rnd>0.5):
                        path.insert(0,front_m_op)
                        item = [item[0], item[1]-1]
                        latest_op_now = front_m_op
                    else:
                        path.insert(0,front_job_op)
                        item = [front_job_m,front_ind]
                        latest_op_now = front_job_op
                        
    
            critical_path.append(path)
        return critical_path
    
    #####統整fitness#####
    def Fitness_func(self, indx, OPS, machine_assign,solution, mo_weight=0.4, iu_weight=0.4, lbs_weight=0.1, otf_weight=0.1):
        # 考慮不同指標的適應度
        move_fit = move_fitness1(solution) # 搬運成本
        max_end_time, job_op_seq, job_finish = Schedule(machine_assign, self.each_machine_time[indx], OPS, self.record_time[indx], self.machine, self.job, self.Find_op_time, self.Job_Op_Num) #結束時間
        inv_utilize_fit = Utilize_rate(self.each_machine_time[indx], max_end_time, self.machine)*100 # 反稼動率
        load_balance_std = Load_balance(self.each_machine_time[indx], self.machine)
        on_time_fit = On_time_fit(self.request_date, job_finish, self.job)*100
        
        # move_fit_N = move_fit_nor(move_fit)
        # inv_utilize_fit_N = utilize_fit_nor(inv_utilize_fit)
        # load_balance_std_N = Load_balance_nor(load_balance_std)
        # on_time_fit_N = On_time_fit_nor(on_time_fit)
        
    
            
        # # 各個粒子適應度開根號減少之間scale的差距
        # total_fitness = (mo_weight*move_fit_N +\
        #                   iu_weight*inv_utilize_fit_N +\
        #                   lbs_weight*load_balance_std_N+\
        #                   otf_weight*on_time_fit_N)
            
        # 不進行正規化
        total_fitness = (mo_weight*move_fit+\
                          iu_weight*inv_utilize_fit +\
                          lbs_weight*load_balance_std+\
                          otf_weight*on_time_fit)/4
            
        # total_fitness = max_end_time
        fitness_array = np.array([{inv_utilize_fit}, {load_balance_std}, {on_time_fit}, {max_end_time}])
        # fitness_array = np.array([{max_end_time}])

        
        return total_fitness, job_op_seq,fitness_array
    
    def Sigmoid(self,x):
        z = np.exp(-x)
        sig = 1 / (1 + z)

        return sig      
    # def Sigmoid(self,x):
    #     z = np.exp(-x)
    #     sig = z / (1 + z)**2
    
    #     return sig             

        
######################################################################################################################
######################################
###############開始使用################
######################################
######################################################################################################################
mk01_data = get_data("MK01.txt")
t = 0
ddr = []
for i in range(1):

    solver = PSOSolver(particle_number=20,machine=6, job=10, iterations =50, cognition_factor=2, \
                       social_factor=2, inertial_factor=0.8, data=mk01_data, Job_Op_Num=Job_Op_Num, request_date=customer_date)
    solver.Initialize()
    ####開始迭代#####
    start = time.time()
    
    best = [] #每次迭代的最佳全局適應度
    particle_obj= [[] for i in range(solver.particle_number)] #每次迭代的各particle適應度
    for iteration in range(solver.iterations):

        if (iteration+1)%1==0:
            print(f"========Iteration {iteration+1}========")

        if(iteration==0):
            solver.Swap_op_seq()
        solver.Move_to_new_positions(iteration)
        solver.Update_best_solution()
        solver.Swap_op_seq()
        best.append(solver.gbest_obj)
        
        if (iteration+1)%1==0:
#             print(f"Job Seq0:{solver.job_seq[0]}")
            print(f"Best fitness:{solver.gbest_obj}")
            print(f"Best job sequence{solver.gbest_OPS}")
            for i in range(solver.particle_number):
    #             print("\n")
    #             print(f"第{i+1}個粒子更新位置:{solver.solutions[i]}")
    #             print(f"第{i+1}個粒子更新後適應度:{solver.compute_fitness(solver.solutions[i])}\n")
    #             print(f"第{i+1}個粒子目前最佳位置與適應度{solver.pbest[i]}:{solver.pbest_obj[i]}\n")
    #             print("------------------------")
                particle_obj[i].append(solver.part_val[i])
    #         print("------------------------")
    #         print("更新後的全局最佳位置與適應度:")
    #         print(f"{solver.gbest}, {solver.gbest_obj}\n")
    
    ddr.append(best[-1])
    print("\n\n")
    # print(f"best:{best}")
    print(f"經過{solver.iterations}次迭代後，最佳適應度為{solver.gbest_obj}")
    print(f"最佳排程為{solver.best_arange}")
    end = time.time()
    total = end-start
    print(f"共訓練{solver.iterations}次:花費{total}秒")
    t+=total

print(f"測試10次平均花費:{t/10}秒")

from matplotlib import pyplot as plt
myset = list(set(ddr))  #myset是另外一個列表，裡面的內容是mylist裡面的無重複 項
height = []
for item in myset:
    height.append(ddr.count(item))
plt.bar(myset, height, color="#6B8F9B", align="center")
plt.ylabel('times')
plt.xlabel('fitness')
plt.xticks(fontsize=12)
p = np.arange(0,len(ddr),1)
plt.yticks(p,fontsize=12)
plt.xticks(range(int(min(ddr)),int(max(ddr)+1),1))
plt.savefig('stability_test.png')


######################################################################################################################
######################################
###############畫圖區################
######################################
######################################################################################################################
#1.畫所有fitness迭代圖
from matplotlib import pyplot as plt

iters = [i+1 for i in range(0,solver.iterations,1)]
fig, ax = plt.subplots(figsize=(20,18))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.title("疊代過程適應度變化",fontsize=20)
plt.xlabel("疊代次數", fontsize=16)
plt.ylabel("適應度", fontsize=16)
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 1,solver.particle_number)]
ax.set_prop_cycle('color', colors)

for i in range(solver.particle_number):
    plt.plot(iters,particle_obj[i],marker="o")
    
p = np.arange(0,np.max(np.max(particle_obj))+100,50)
plt.yticks(p,fontsize=12)
plt.xticks(range(1,solver.iterations,50))

name = []
for i in range(solver.particle_number):
    name.append("particle_"+str(i+1))
plt.legend(name, loc='upper right',fontsize=12)
plt.savefig('iter_history.png')

#2.畫全局最佳fitness迭代圖
plt.figure(figsize=(12,9))
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.title("疊代過程之最佳適應度變化",fontsize=20)
plt.xlabel("疊代次數", fontsize=16)
plt.ylabel("適應度", fontsize=16)
plt.xticks(fontsize=12)
p = np.arange(0,best[0]+20,10)
plt.yticks(p,fontsize=12)
plt.xticks(range(1,solver.iterations,50))
iters = [i+1 for i in range(solver.iterations)]

plt.plot(iters,best,color="#008B8B",marker="o")
plt.savefig('best_fitness.png')




#3.畫排程甘特圖
#設定顏色
colors = ["#F08080","#6495ED","#8FBC8F","#FFA500","#CCBBAA","#6B8F9B","#008B8B","#FFD700","#696969","#9370DB"]
#畫柱狀圖
bar_width=0.6
plt.figure(figsize=(12,9))
for i in range(solver.machine):
    for j in range(len(solver.gbest_time_seq[i])):
        k = solver.best_job_op_seq[i][j][0] #第幾張工單
        w = solver.gbest_time_seq[i][j][1]-solver.gbest_time_seq[i][j][0] #時間差
        img = plt.barh(y=i, width=w, color=colors[k],left=solver.gbest_time_seq[i][j][0],alpha=0.8, height=bar_width, 
                        align='center', edgecolor = 'k', linewidth=1.5, linestyle='-')
        locals()["job"+str(k)+"_legend"] = img


#在柱狀圖上顯示具體數值, ha參數控制水平對齊方式{'center','top','bottom','baseline','center_baseline'}, va控制垂直對齊方式({'center','right','left'})
for y, x in enumerate(solver.gbest_time_seq):
    if x!=[]:
        plt.text(x[-1][-1]+1, y, '%s' % x[-1][-1], ha='left', va='center',fontsize=15,fontweight="extra bold")


plt.xticks(range(0,int(45+15),5),fontsize=12)
plt.yticks(fontsize=12)

# 设置标题
plt.title("Flexible Job Shop Scheduling",fontsize=20)
# 为两条坐标轴设置名称
plt.xlabel("Time Sequence",fontsize=16)
plt.ylabel("Machine",fontsize=16)

variable = []
for i in range(solver.job):
    variable.append(locals()["job"+str(i)+"_legend"])
    
name = []
for i in range(solver.job):
    name.append("job"+str(i+1))

# 显示图例
plt.legend(variable,name,fontsize=15)
plt.gca().invert_yaxis() #將Y軸顛倒
plt.savefig('Gantt chart.png')
