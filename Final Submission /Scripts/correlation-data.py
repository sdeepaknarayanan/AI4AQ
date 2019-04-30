
# coding: utf-8

# In[10]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:16:55 2019

@author: deepak
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
import matplotlib

    
df = pd.read_csv('../../../Datasets/Updated_Delhi_Scaled.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


def max_variance_sampling(pool_df, learners):
    std_dev = pd.DataFrame()
    for station in station_pool:
        new_df = pool_df.groupby('Station').get_group(station)
        temp = {}
        for i in learner_list:
            temp[i] = learner_list[i].predict(new_df[['Latitude','Longitude','Date']])
        temp = pd.DataFrame(temp)
        std_dev[station] = [(temp.std(axis=1).mean())]
    return std_dev.loc[0].idxmax()

def update_pool_train(station, df_main, x_train, y_train, pool_df,day):
    '''
        Add data completely - query whenever needed using the day as an index
        TIll that day from the start
        
        Create a dictionary with station names as keys and added days as values'''
    station_to_add = station
    train_to_add_1 = df_main[df_main['Station']==station_to_add][day:day+1]
    train_to_add_2 = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_train])
    x_train = np.append(x_train,train_to_add_2[['Latitude','Longitude','Date']],axis = 0)
    x_train =  np.append(x_train, train_to_add_1[['Latitude','Longitude','Date']],axis=0)
    y_train = np.append(y_train, train_to_add_1['PM2.5'])
    y_train = np.append(y_train, train_to_add_2['PM2.5'])
    station_train.append(station_to_add)
    pool_df = pool_df[pool_df['Station']!=station_to_add]
    station_pool.remove(station_to_add)
    pool_to_add = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
    pool_df = pd.concat([pool_df,pool_to_add])
    return x_train, y_train, pool_df

def daily_update(df_main, x_train, y_train, pool_df,day):
    train_to_add_2 = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_train])
    x_train = np.append(x_train,train_to_add_2[['Latitude','Longitude','Date']],axis = 0)
    y_train = np.append(y_train, train_to_add_2['PM2.5'])
    pool_to_add = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
    pool_df = (pd.concat([pool_df,pool_to_add]))
    return x_train, y_train, pool_df


initial_number_of_days = 15


station_list = list(df['Station'].unique())


station_numbering = {}
for i in range(len(station_list)):
    station_numbering[station_list[i]] = 'S'+str(i+1)

for i in range(len(station_list)):
    test_set = [station_list[i]]
    station_test = deepcopy(test_set)

    
    
    new_station_list = deepcopy(station_list)
    new_station_list.remove(station_list[i])
    train_set = new_station_list[:4]
    pool_set = new_station_list[4:]
    
    station_train = deepcopy(train_set)
    station_pool = deepcopy(pool_set)
    
    
    train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    x_train = train[['Latitude','Longitude','Date']]
    y_train = train['PM2.5']
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
    Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
    x_pool = Pool[['Latitude','Longitude','Date']]
    y_pool = Pool['PM2.5']
    x_pool = np.array(x_pool)
    y_pool = np.array(y_pool)
    
    Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
    Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
    
    x_test = Test[['Latitude','Longitude','Date']]
    y_test = Test['PM2.5']
    #x_test = np.array(x_test)
    y_test = np.array(y_test)
    #pd.DataFrame(y_test).to_csv('Data/Ground Truth/'+station_list[i]+'.csv')
    
    learner_list = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
    for i in learner_list:
        learner_list[i].fit(x_train,y_train)
    mae = np.zeros(5)
    for i in range(5):
        mae[i] = mean_absolute_error(learner_list[i+1].predict(x_test),y_test)
    print('The initial MAE Mean is ',mae.mean())
    start = mae.mean()
    
    
    
    day = initial_number_of_days
    active_learning = np.zeros(32)
    active_learning[0] = start
    day_wise_prediction = {i:0 for i in range(15,46)}
    for itr, day in enumerate(range(15,46)):
        #print(itr)
        if itr%5==0:
            pool_cpy = Pool.copy()
            station_to_add = max_variance_sampling(Pool,learner_list)
            x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
        else:
            x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
        prediction_itr = []
        mae = []
        for i in learner_list:
            learner_list[i].fit(x_train, y_train)
            prediction = learner_list[i].predict(x_test)
            prediction_itr.append(prediction)
            mae.append(mean_absolute_error(prediction, y_test))
        mean_values = np.array(prediction_itr).mean(axis=0)
        day_wise_prediction[day] = mean_values
        del mean_values
        active_learning[itr+1]=np.mean(mae)
    to_csv_df = pd.DataFrame(day_wise_prediction)
    to_csv_df.to_csv('Predicted4/'+test_set[0]+'.csv')
