# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 08:51:17 2021

@author: jerry
"""
import pandas as pd
import numpy as np



# 初始排程的搬運成本
def move_fitness1(solutions,jobs):
    # 將搬運成本表轉換為字典
    path = r"Mk01搬運成本表.xlsx"
    dataTable = np.array(pd.read_excel(path, index_col=None, engine='openpyxl'))
    dataDict = {}
    for i in range(dataTable.shape[0]):
        for j in range(dataTable.shape[1]):
            dataDict['{0}{1}'.format(i, j)] = dataTable[i][j] 
            
    # 平均計算每個job的機台搬運成本之總和
    fitness = 0
    for i in range(len(solutions)):
        start = solutions[i][0].tolist().index(1)
        for j in range(1, len(solutions[i])):
            end = solutions[i][j].tolist().index(1)
            fitness += dataDict['{0}{1}'.format(start, end)]
            start = end

    return float(fitness)/jobs

# 機台故障的搬運成本
def move_fitness2(before_assignment, after_assignment):
    def array_merge(data):
        newdata = data[0]
        for i in range(1, len(data)):
            newdata = np.append(newdata, data[i], axis=0)
        return newdata
    before_assignment = array_merge(before_assignment)
    after_assignment = array_merge(after_assignment)
    path = r"Mk01搬運成本表.xlsx"
    dataTable = np.array(pd.read_excel(path, index_col=None, engine='openpyxl'))
    dataDict = {}
    for i in range(dataTable.shape[0]):
        for j in range(dataTable.shape[1]):
            dataDict['{0}{1}'.format(i, j)] = dataTable[i][j] 
    fitness = 0
    move_matrix = after_assignment-before_assignment
    for i in range(move_matrix.shape[0]):
        try:
            a = move_matrix[i].tolist().index(-1)
            b = move_matrix[i].tolist().index(1)
            fitness += dataDict['{0}{1}'.format(a, b)]
        except:
            pass
    return fitness


# 排程最晚結束時間
def Schedule(op_machine_pair, each_machine_time, OPS, record_time, machine, job, find_op_time, job_op_num):
    prev_time = [0 for _ in range(job)] #紀錄每一工單中當前最晚的完成時間
    job_accu = [0 for i in range(job)] #紀錄當前job排到第幾個OP
    ma_max_time = [0 for i in range(machine)] #每個機台目前最晚的結束時間
    job_op_seq = [[] for i in range(machine)]
    job_finish = np.zeros(job)

    for i in range(len(OPS)):
        op_num = job_accu[OPS[i]] #該job的第幾道OP
        m = op_machine_pair[OPS[i]][op_num] #第幾機台
        period = find_op_time([OPS[i],op_num],m) #該OP所花時間
        job_op_seq[m].append([OPS[i], op_num])
        if (prev_time[OPS[i]]>ma_max_time[m]):
            each_machine_time[m].append([prev_time[OPS[i]], prev_time[OPS[i]]+period])
            ST = round(prev_time[OPS[i]],2)
            ET = round(prev_time[OPS[i]]+period,2)
            record_time[m].append({"Job":OPS[i], "Op":op_num, "ST":ST, "ET":ET})
            prev_time[OPS[i]]+=period
            ma_max_time[m] = prev_time[OPS[i]]
        else:
            each_machine_time[m].append([ma_max_time[m], ma_max_time[m]+period])
            ST = round(ma_max_time[m],2)
            ET = round(ma_max_time[m]+period,2)
            record_time[m].append({"Job":OPS[i], "Op":op_num, "ST":ST, "ET":ET})
            ma_max_time[m]+=period
            prev_time[OPS[i]] = ma_max_time[m]
        
        if(op_num==job_op_num[OPS[i]]-1): #若該製程為該工單最後一個製程
            job_finish[OPS[i]]=prev_time[OPS[i]]  #紀錄該結束時間
        job_accu[OPS[i]]+=1 #將OP序號加1
    # best_time = round(np.max(prev_time),2)
    best_time = np.max(prev_time)
    return best_time, job_op_seq, job_finish  #返回所有工單中最晚的完成時間 & 工單時間計算順序

# 平均稼動率(用反指標做)
def Utilize_rate(each_machine_time, makespan, machine):
    counter_rate = 0
    for i in range(machine):
        usage_time = 0
        if(each_machine_time[i]!=[]):
            for item in each_machine_time[i]:
                usage_time+=item[1]-item[0]
            machine_rate = (usage_time)/makespan 
        else:
            machine_rate = 0
        counter_rate+=(1-machine_rate)
    avg_rate = counter_rate/machine
    return avg_rate

# 負載平衡
def Load_balance(each_machine_time, machine):
    machine_load = []
    for i in range(machine):
        usage_time = 0
        if(each_machine_time[i]!=[]):
            for item in each_machine_time[i]:
                usage_time+=item[1]-item[0]
        machine_load.append(usage_time)
    avg_std = np.std(machine_load)
    return avg_std

# 準交率
def On_time_fit(req_date, real_date, job):
    total = 0
    count = 0
    for i in range(job):
        diff = real_date[i]-req_date[i]
        if(diff>0): #若有遲交情形發生，才加上遲交時間長度
            # total+=diff
            count+=1
    return float(count)/job

# 搬運成本適應度之標準化
def move_fit_nor(data):
    data_avg = 138.46
    data_std = 14.52
    
    normalized_data = (data - data_avg)/(data_std) 
    
    # if normalized_data < 0:
    #     normalized_data = 0
    # elif normalized_data > 1:
    #     if normalized_data > 10:
    #         pass
    #     else:
    #         normalized_data = 1
            
    return normalized_data
            
# 稼動率適應度之正規化
def utilize_fit_nor(data):
    data_avg = 38.52
    data_std = 4.55
    
    normalized_data = (data - data_avg)/(data_std)  
    
    # if normalized_data < 0:
    #     normalized_data = 0
    # elif normalized_data > 1:
    #     if normalized_data > 10:
    #         pass
    #     else:
    #         normalized_data = 1
    
    return normalized_data

# 負載平衡適應度之正規化
def Load_balance_nor(data):
    data_avg = 24.93
    data_std = 4.56
    
    normalized_data = (data - data_avg)/(data_std)  
    
    # if normalized_data < 0:
    #     normalized_data = 0
    # elif normalized_data > 1:
    #     if normalized_data > 10:
    #         pass
    #     else:
    #         normalized_data = 1
    
    return normalized_data


# 準交率適應度之正規化
def On_time_fit_nor(data):
    data_avg = 50.0
    data_std = 0.5
    
    # data_avg = 355.34
    # data_std = 87.91
    
    normalized_data = (data - data_avg)/(data_std)  
    
    # if normalized_data < 0:
    #     normalized_data = 0
    # elif normalized_data > 1:
    #     if normalized_data > 10:
    #         pass
    #     else:
    #         normalized_data = 1
    
    return normalized_data