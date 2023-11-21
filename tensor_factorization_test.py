#imported packages
#!pip install tensorly

#imported packages
import os
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import non_negative_tucker
import math
from scipy import stats
import csv
from matplotlib.pyplot import figure
import matplotlib
from sklearn.metrics import mean_squared_error as rmse
import time

#defined functions
#data column methods

#convert data to array
def getseqid(targetval):
    #convert taz idx
    targetidx = {}
    idxtarget = {}
    idx = 0
    for i in targetval:
        targetidx[i] = idx        
        idxtarget[idx] = i
        idx += 1
    return(targetidx, idxtarget)

def data2arr(datapath, binaryidx=True, to2d=True):
    taz2idx = lambda x: tazidx[x]
    str2time = lambda x: int(int(x.split(':')[0])*12 + int(x.split(':')[1])/5)
    timelen = 24*12 #with 5-min interval
    newid2idx = lambda x: newididx[x]
    
    datapath = raw_dt_path[0]
    dt = pd.read_csv(datapath)
    
    #convert time and location
    tazidx, idxtaz = getseqid(np.unique(dt['taz']))
    dt['tazidx'] = dt['taz'].apply(taz2idx)
    dt['timeidx'] = dt['time'].apply(str2time)

    newididx, idxnewid = getseqid(np.unique(dt['newid']))
    dt['newididx'] = dt['newid'].apply(newid2idx)
    
    if to2d == True:
        dt['ltidx'] = dt['tazidx']*timelen+dt['timeidx']
        nrow = len(newididx)
        ncol = timelen * len(tazidx)

        arr = np.zeros((nrow, ncol))
        dt = dt[['newididx','ltidx', 'sum']].to_numpy()

        #assign binary index
        for i, j, count in dt:
            if binaryidx == True:
                arr[i, j] = 1
            else:
                arr[i, j] = count
        return(arr)
    else:
        nrow = len(newididx)
        ntaz = len(tazidx)
        ntime = timelen
        
        arr = np.zeros((nrow, ntaz, ntime))
        dt = dt[['newididx', 'tazidx', 'timeidx', 'sum']].to_numpy()
        
        for i, j, k, count in dt:
            if binaryidx == True:
                arr[i, j, k] = 1
            else:
                arr[i, j, k] = sum
        return(arr)
    
#process the fcd data
path = "G:/My Drive/2021/Bias/sumo_simulation/"
#os.listdir()

raw_dt_path = []
for i in os.listdir(path):
    if len(i) == 12 and '2017' in i:
        raw_dt_path.append(path+i)
        
arr = data2arr(raw_dt_path[0], to2d=False)



rank_individuals = 10
rank_locations = 10
rank_times = 12

#tensor factorization
starttime = time.time()
core, factors = tucker(arr, rank=[rank_individuals, rank_locations, rank_times])
endtime = time.time()
runningtime_tf = endtime-starttime
print(f'running time tf: {runningtime_tf}')
reconstructed_data = tl.tucker_to_tensor((core, factors)) 
rmse_score = rmse(arr, reconstructed_data, squared=False)
mae_score = rmse(arr, reconstructed_data, squared=True)
print(f'rmse tf ={rmse_score}')
write_contents_tf = [rank_individuals, rank_locations, rank_times, runningtime_tf, rmse_score, mae_score, 'tf']

#non negative 
starttime = time.time()
core, factors = non_negative_tucker(arr, rank=[rank_individuals, rank_locations, rank_times])
endtime = time.time()
runningtime_nntf = endtime-starttime
reconstructed_data = tl.ttensors_to_tensor(core, factors)
rmse_score = rmse(arr, reconstructed_data, squared=False)
mae_score = rmse(arr, reconstructed_data, squared=True)
print(f'rmse nntf ={rmse_score}')
print(f'running time nntf: {runningtime_nntf}')
write_contents_nntf = [rank_individuals, rank_locations, rank_times, runningtime_nntf, rmse_score, mae_score, 'nntf']

print(write_contents_tf)
print(write_contents_nntf)
with open('G:/My Drive/2021/Bias/tf_results.txt', 'a') as f:
    f.write(f'{write_contents_tf}\n')
    f.write(f'{write_contents_nntf}\n')
#def main():