#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:34:29 2019

@author: deepak
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
import matplotlib

    
# df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')
## INCLUDE DIRECTORY OF THE DATASET
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

from math import sqrt
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax




initial_number_of_days = 15


station_list = list(df['Station'].unique())


station_numbering = {}
for i in range(len(station_list)):
    station_numbering[station_list[i]] = 'S'+str(i+1)

queried = {i:{} for i in station_list}
for i in range(len(station_list)):
    '''if station_list[i]!='R K Puram':
        continue'''
    print(station_list[i])
    print('*'*100)
    test_set = [station_list[i]]
    new_station_list = deepcopy(station_list)
    new_station_list.remove(station_list[i])
    train_set = new_station_list[:4]
    pool_set = new_station_list[4:]
    
    station_train = deepcopy(train_set)
    station_test = deepcopy(test_set)
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
    x_test = np.array(x_test)
    y_test = np.array(y_test)
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
    
    for itr, day in enumerate(range(15,46)):
        #print(itr)
        if itr%5==0:
            pool_cpy = Pool.copy()
            station_to_add = max_variance_sampling(Pool,learner_list)
            queried[station_test[0]][itr]  = station_to_add
            print(station_to_add,'\n')
            x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
        else:
            x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
        prediction = []
        mae = []
        for i in learner_list:
            learner_list[i].fit(x_train, y_train)
            prediction = learner_list[i].predict(x_test)
            mae.append(mean_absolute_error(prediction, y_test))
        active_learning[itr+1]=np.mean(mae)
    
    
    del station_train 
    del station_test
    del station_pool
    plt.plot('*'*60)
    # continue
    random2 = {i:{j:0 for j in range(31)} for i in range(1,51)}
    random_final = np.zeros(32)
    seed = [i+1 for i in range(50)]
    start_days = 15
    abs_mean = {}
    for elem in seed:
        
        station_train = deepcopy(train_set)
        station_test = deepcopy(test_set)
        station_pool = deepcopy(pool_set)
        print(len(station_train),len(station_test),len(station_pool))
        np.random.seed(elem)
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
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        
        learner_list = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
        for i in learner_list:
            learner_list[i].fit(x_train,y_train)
        mae = np.zeros(5)
        for i in range(5):
            mae[i] = mean_absolute_error(learner_list[i+1].predict(x_test),y_test)
        random_final[0] = (np.mean(mae))
        
        for itr,day in enumerate(range(15,46)):
            if itr%5==0:
                station_to_add = np.random.choice(station_pool)
                x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
            else:
                x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
            day+=1
            mae = []
            for i in learner_list:
                learner_list[i].fit(x_train, y_train)
                prediction = (learner_list[i].predict(x_test))
                mae.append(mean_absolute_error(prediction, y_test))
            abs_mean[itr] = np.mean(mae)
            random2[elem][itr] = np.mean(mae)
            
    random_final[1:] = np.array(pd.DataFrame(random2).mean(axis = 1))
    stddev = np.array(pd.DataFrame(random2).std(axis=1))
    stddev = np.append(np.array([0]),stddev)
    latexify(columns=1)
    plt.plot(active_learning,marker = '*',label='Active Sampling')
    for j in range(1,31,5):
        plt.axvline(j,ls='--',color='k',lw=0.5)
    format_axes(plt.gca())
    plt.errorbar([i for i in range(32)],random_final,yerr=stddev,marker='.',label='Random Sampling')
    #plt.title('Estimation at '+station_numbering[station_test[0]])
    '''
    plt.xlabel('Number of Days Elapsed')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='best', shadow=False)
    plt.tight_layout()
    plt.savefig('Final_Plots_4_marker/'+station_numbering[station_test[0]]+'.pdf', dpi=300)
    plt.close()'''
    