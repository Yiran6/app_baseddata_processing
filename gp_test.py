#import packages
import pandas as pd
import numpy as np
from datetime import *
from collections import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import rcParams
import xlrd as xl
import csv
import os
import math
from scipy import stats
import csv
from matplotlib.pyplot import figure
import matplotlib
from sklearn.metrics import mean_squared_error as rmse
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct
import seaborn as sns
from scipy import stats
import time

#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, Matern, WhiteKernel, ExpSineSquared, RationalQuadratic
import warnings

#calculate the distance
#!pip install haversine
from haversine import haversine, Unit

from sklearn.preprocessing import StandardScaler

#source code
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

#defined functions
def app_data_processing(datapath, gettazidx=False):
    dt = pd.read_csv(datapath)
    #convert idx
    tazidx = {}
    idxtaz = {}
    idx = 0
    for i in np.unique(dt['taz']):
        tazidx[i] = idx        
        idxtaz[idx] = i
        idx += 1
    taz2idx = lambda x: tazidx[x]

    dt['tazidx'] = dt['taz'].apply(taz2idx)

    str2time = lambda x: int(int(x.split(':')[0])*12 + int(x.split(':')[1])/5)
    dt['timeidx'] = dt['time'].apply(str2time)
    
    groupdt = dt.groupby(by=['tazidx','timeidx']).size().to_frame()
    groupdt.reset_index(inplace=True)
    groupdt = groupdt.to_numpy()
    
    # write2array
    nrow = len(np.unique(dt['tazidx']))
    ncol = len(np.unique(dt['timeidx']))
    #create the array
    arr = np.zeros((nrow, ncol))
    for taz_id, time_id, count in groupdt:
        arr[taz_id, time_id]  = count
    if gettazidx == True:
        return(arr, idxtaz, tazidx)
    else:
        return(arr)
    
def add_time_series(X, scale_idx):
    time_series = np.arange(1, (len(X)+1))
    
    X_with_time = np.column_stack((X, time_series))
    #standardize the input features
    if scale_idx==True:
        scaler = StandardScaler()
        X_with_time = scaler.fit_transform(X_with_time)
    return(X_with_time)

def check_kernel4GP_timeseries(nrow1, nrow2, idxtaz, arr1, arr2, n_restart=50, plot_index=False, scale_idx=False):
    #print(data_test.loc[nrow1][0]+' vs. '+data_test.loc[nrow2][0])
    
    #print('Pearson correlation: ', pc )

    #get average
    X1_ = arr1[nrow1]
    X2_ = arr1[nrow2]
    pc = stats.pearsonr(X1_, X2_)

    X1_test = arr2[nrow1]
    X2_test = arr2[nrow2]    
    
    #np.random.seed(0)
    
    if np.sum(X1_) != 0 and np.sum(X1_test) != 0 and np.sum(X2_)!=0 and np.sum(X2_test)!=0:
        X_with_time = add_time_series(X1_, scale_idx)
        X1_test_with_time = add_time_series(X1_test, scale_idx)
        return(conduct_kernel(nrow1, nrow2, idxtaz, X_with_time, X2_, X1_test_with_time, X2_test, n_restart, pc, plot_index, ts_idx=True))
    else:
        return([0,0,0])
    
# Generate synthetic data
def check_kernel4GP(nrow1, nrow2, idxtaz, arr1, arr2, n_restart=50, plot_index=False):
    #print(data_test.loc[nrow1][0]+' vs. '+data_test.loc[nrow2][0])
    
    #print('Pearson correlation: ', pc )
    
    #get average
    X1_ = arr1[nrow1]
    X2_ = arr1[nrow2]
    pc = stats.pearsonr(X1_, X2_)

    X1_test = arr2[nrow1]
    X2_test = arr2[nrow2]    
    
    
    if np.sum(X1_) != 0 and np.sum(X1_test) != 0 and np.sum(X2_)!=0 and np.sum(X2_test)!=0:
        X1_ = X1_.reshape(-1, 1)
        X1_test = X1_test.reshape(-1, 1)
        return(conduct_kernel(nrow1, nrow2, idxtaz, X1_, X2_, X1_test, X2_test, n_restart, pc, plot_index, ts_idx=False))
    else:
        return([0,0,0])
    
def conduct_kernel(nrow1, nrow2, idxtaz, X, y, X_test, X2_test, 
                   n_restart, pc, plot_index, ts_idx):
    runningtime = []
    start_time = time.time()
    # Linear Kernel (Dot-Product Kernel)
    linear_kernel = ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
    gp_linear = GaussianProcessRegressor(kernel=linear_kernel, n_restarts_optimizer=n_restart)
    gp_linear.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)

    # Radial Basis Function (RBF) Kernel (Squared Exponential Kernel)
    start_time = time.time()
    rbf_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp_rbf = GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=n_restart)
    gp_rbf.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)
    
    # Matérn Kernel
    start_time = time.time()
    matern_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    gp_matern = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=n_restart)
    gp_matern.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)

    # Constant Kernel (White Kernel)
    start_time = time.time()
    constant_kernel = ConstantKernel(1.0) * WhiteKernel(noise_level=0.1)
    gp_constant = GaussianProcessRegressor(kernel=constant_kernel, n_restarts_optimizer=n_restart)
    gp_constant.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)

    # Polynomial Kernel
    # polynomial_kernel = ConstantKernel(1.0) * DotProduct(sigma_0=1.0, degree=2)
    # gp_polynomial = GaussianProcessRegressor(kernel=polynomial_kernel, n_restarts_optimizer=10)
    # gp_polynomial.fit(X, y)

    # Periodic Kernel (ExpSineSquared)
    start_time = time.time()
    periodic_kernel = ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
    gp_periodic = GaussianProcessRegressor(kernel=periodic_kernel, n_restarts_optimizer=n_restart)
    gp_periodic.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)

    # Rational Quadratic Kernel
    start_time = time.time()
    rq_kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=0.1)
    gp_rq = GaussianProcessRegressor(kernel=rq_kernel, n_restarts_optimizer=n_restart)
    gp_rq.fit(X, y)
    end_time = time.time()
    runningtime.append(end_time-start_time)
    model_name = ['linear', 'rbf', 'matern', 'white','periodic', 'rational quadratic']
    
    y_pred_linear_ori = gp_linear.predict(X, return_std=True)[0]
    y_pred_rbf_ori = gp_rbf.predict(X, return_std=True)[0]
    y_pred_matern_ori = gp_matern.predict(X, return_std=True)[0]
    y_pred_constant_ori = gp_constant.predict(X, return_std=True)[0]
    #y_pred_polynomial, sigma_polynomial = gp_polynomial.predict(x_pred, return_std=True)
    y_pred_periodic_ori = gp_periodic.predict(X, return_std=True)[0]
    y_pred_rq_ori = gp_rq.predict(X, return_std=True)[0]
    rmse_org = np.array([rmse(y, y_pred_linear_ori, squared=False), 
             rmse(y, y_pred_rbf_ori, squared=False), 
             rmse(y, y_pred_matern_ori, squared=False), 
             rmse(y, y_pred_constant_ori, squared=False), 
             rmse(y, y_pred_periodic_ori, squared=False), 
             rmse(y, y_pred_rq_ori, squared=False)])  
    #print(rmse_org)
    #print('best rmse, train data vs pred, model='+model_name[np.argmin(rmse_org)]+', rmse='+str(np.min(rmse_org)))
    
    #X_test = X1_test.reshape(-1, 1)
    
    y_pred_linear_test = gp_linear.predict(X_test, return_std=True)[0]
    y_pred_rbf_test = gp_rbf.predict(X_test, return_std=True)[0]
    y_pred_matern_test = gp_matern.predict(X_test, return_std=True)[0]
    y_pred_constant_test = gp_constant.predict(X_test, return_std=True)[0]
    #y_pred_polynomial, sigma_polynomial = gp_polynomial.predict(x_pred, return_std=True)
    y_pred_periodic_test = gp_periodic.predict(X_test, return_std=True)[0]
    y_pred_rq_test = gp_rq.predict(X_test, return_std=True)[0]
    rmse_test = np.array([rmse(X2_test, y_pred_linear_test, squared=False), 
             rmse(X2_test, y_pred_rbf_test, squared=False), 
             rmse(X2_test, y_pred_matern_test, squared=False), 
             rmse(X2_test, y_pred_constant_test, squared=False), 
             rmse(X2_test, y_pred_periodic_test, squared=False), 
             rmse(X2_test, y_pred_rq_test, squared=False)]) 
    #print(rmse_test)
    #print('best rmse, test data vs pred, model='+model_name[np.argmin(rmse_test)]+', rmse='+str(np.min(rmse_test)))
    

    # Plot the results
    if plot_index == True:
        # Define test points for prediction
        #if ts_idx == False:
        #    x_pred =  np.arange(1, max(X_test)+1, 1).reshape(-1, 1)
        #else:
        #    x_pred = add_time_series_pred(X_test)
        x_pred = X_test

        # Make predictions using each GP model
        y_pred_linear, sigma_linear = gp_linear.predict(x_pred, return_std=True)
        y_pred_rbf, sigma_rbf = gp_rbf.predict(x_pred, return_std=True)
        y_pred_matern, sigma_matern = gp_matern.predict(x_pred, return_std=True)
        y_pred_constant, sigma_constant = gp_constant.predict(x_pred, return_std=True)
        #y_pred_polynomial, sigma_polynomial = gp_polynomial.predict(x_pred, return_std=True)
        y_pred_periodic, sigma_periodic = gp_periodic.predict(x_pred, return_std=True)
        y_pred_rq, sigma_rq = gp_rq.predict(x_pred, return_std=True)
        
        plt.figure(figsize=(16, 10))
        #https://matplotlib.org/stable/gallery/color/named_colors.html
        plt.subplot(4, 2, 1)
        if ts_idx == True:
            x_pred = x_pred[:, 0]
            X = X[:, 0]
        plt.title(str(idxtaz[nrow1])+' vs. '+str(idxtaz[nrow2]))
        plt.plot(x_pred, y_pred_linear, c='orange', label='linear')
        plt.plot(x_pred, y_pred_rbf, c='blue', label='rbf')
        plt.plot(x_pred, y_pred_matern, c='black', label='matern')
        plt.plot(x_pred, y_pred_constant, c='red', label='white')
        plt.plot(x_pred, y_pred_periodic, c='cyan', label='periodic')
        plt.plot(x_pred, y_pred_rq, c='darkcyan', label='rational quadratic')
        plt.scatter(X, y, c='midnightblue', s=30, label='Data')

        #plt.fill_between(x_pred.ravel(), y_pred_linear - 1.96 * sigma_linear, y_pred_linear + 1.96 * sigma_linear, alpha=0.2, color='k')
        #plt.title('Linear Kernel')

        # Repeat the above plotting for other kernels (RBF, Matern, Constant, Polynomial, Periodic, Rational Quadratic)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    results = [idxtaz[nrow1], idxtaz[nrow2], pc[0], model_name[np.argmin(rmse_org)], np.min(rmse_org), model_name[np.argmin(rmse_test)], np.min(rmse_test), 
              (np.max(rmse_org)-np.min(rmse_org)), (np.max(rmse_test)-np.min(rmse_test))]
    results.extend(rmse_org)
    results.extend(rmse_test)
    results.extend(runningtime)
    return(results)
    
def change_timeinterval(time_interval): #in min
    fold = time_interval/5 #previsou is 5min interval
    return(fold)
    
def get_data_newinterval(time_interval, origin_arr):
    fold = change_timeinterval(time_interval)
    ntaz, ntime = origin_arr.shape
    return(np.mean(origin_arr.reshape(ntaz, int(ntime/fold), int(fold)),axis=2))


#process the fcd data
path = "G:/My Drive/2021/Bias/sumo_simulation/"
#os.listdir()

raw_dt_path = []
for i in os.listdir(path):
    if len(i) == 12 and '2017' in i:
        raw_dt_path.append(path+i)
        
#data processing
#for app-based data
arr04042017, idxtaz, tazidx = app_data_processing(raw_dt_path[0], True)
arr04052017 = app_data_processing(raw_dt_path[1])

#for taz data
#get distance based on taz centroids
gp_path = "G:\\My Drive\\2021\\Gaussianprocess\\"
taz_latlon = pd.read_csv(gp_path+'taz_centroids.csv')

#select a taz, calculate the distance between centroid and sorted the taz based on distance
selected_loc = 510
lat0, lon0 = taz_latlon[['lat','lon']][taz_latlon['TAZ']==selected_loc].values[0]
taz_dist = []
for taz, lat, lon in taz_latlon[['TAZ', 'lat', 'lon']].values:
    if taz != selected_loc:
        dist = haversine((lat0, lon0), (lat, lon), unit='mi')
        taz_dist.append([int(taz), dist])
        
taz_latlon = pd.DataFrame(taz_dist)

taz_latlon = pd.DataFrame(taz_dist)
taz_latlon.columns = ['taz', 'dist2'+str(selected_loc)]
taz_latlon = taz_latlon.sort_values(by=['dist2510'])
taz_latlon.reset_index(inplace=True)
taz_latlon.drop('index', inplace=True, axis=1)

time_interval = 5
arr1 = get_data_newinterval(time_interval, arr04042017)
arr2 = get_data_newinterval(time_interval, arr04052017)

other_tazs_sorted = np.ndarray.flatten(taz_latlon[['taz']].values)

#def check_kernel4GP_timeseries(nrow1, nrow2, idxtaz, arr1, arr2, n_restart=50, plot_index=False, scale_idx=False)
#check_kernel4GP(nrow1, nrow2, idxtaz, arr1, arr2, n_restart=50, plot_index=False)
wo_ts_test = []
for i in other_tazs_sorted:
    cur_results = check_kernel4GP(tazidx[selected_loc], tazidx[i], idxtaz, arr1, arr2, 50, False)
    if len(cur_results) == 3:
        pass
    else:
        wo_ts_test.append(cur_results)

print('\n ******with time series******')
ts_test = []
for i in other_tazs_sorted:
    cur_results = check_kernel4GP_timeseries(tazidx[selected_loc], tazidx[i], idxtaz, arr1, arr2, 50, False, False)
    if len(cur_results) == 3:
        pass
    else:
        ts_test.append(cur_results)
        
wo_ts_test = pd.DataFrame(wo_ts_test)

wo_ts_test.columns = ['taz1', 'taz2', 'p_cor', 'train_best', 'rmse_val', 'test_best', 'rmse_val_test','diff_train', 'diff_test',
                         'linear_rmse_o', 'rbf_rmse_o', 'matern_rmse_o', 'white_rmse_o','periodic_rmse_o', 'rational quadratic_rmse_o',
                         'linear_rmse_t', 'rbf_rmse_t', 'matern_rmse_t', 'white_rmse_t','periodic_rmse_t', 'rational quadratic_rmse_t',
                         'linear_time', 'rbf_time', 'matern_time', 'white_time','periodic_time', 'rational quadratic_time']
print(wo_ts_test.describe(include='all'))

ts_test = pd.DataFrame(ts_test)

ts_test.columns = ['taz1', 'taz2', 'p_cor', 'train_best', 'rmse_val', 'test_best', 'rmse_val_test','diff_train', 'diff_test',
                         'linear_rmse_o', 'rbf_rmse_o', 'matern_rmse_o', 'white_rmse_o','periodic_rmse_o', 'rational quadratic_rmse_o',
                         'linear_rmse_t', 'rbf_rmse_t', 'matern_rmse_t', 'white_rmse_t','periodic_rmse_t', 'rational quadratic_rmse_t',
                         'linear_time', 'rbf_time', 'matern_time', 'white_time','periodic_time', 'rational quadratic_time']
print('\n ******with time series******')
print(ts_test.describe(include='all'))

wo_ts_test.to_csv(f'{gp_path}wo_ts_test_w{str(time_interval)}min.csv')
ts_test.to_csv(f'{gp_path}ts_test_w{str(time_interval)}min.csv')