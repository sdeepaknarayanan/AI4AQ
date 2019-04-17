#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:40:39 2019

@author: deepak
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy

df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')
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


station_mapping = {
        'Anand Vihar':1,
        'Aya Nagar, Delhi - IMD':2,
        'Burari Crossing, Delhi - IMD':3,
        'CRRI Mathura Road, Delhi - IMD':4,
        'Delhi Technological University':5,
        'IGI Airport Terminal-3, Delhi - IMD':6,
        'IHBAS':7,
        'Income Tax Office':8,
        'Lodhi Road, Delhi - IMD':9,
        'Mandir Marg':10,
        'NSIT Dwarka':11,
        'North Campus, Delhi - IMD':12,
        'Punjabi Bagh':13,
        'Pusa, Delhi - IMD':14,
        'R K Puram':15,
        'Shadipur':16,
        'Siri Fort':17
        }

station_list = list(station_mapping.keys())

inverse_mapping = {
        station_mapping[i]:i for i in station_list}

"""
To DO - for all stations"""
station_train = ['NSIT Dwarka','Delhi Technological University','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['Anand Vihar']
station_pool  = ['North Campus, Delhi - IMD', 'R K Puram',
       'Pusa, Delhi - IMD', 'Aya Nagar, Delhi - IMD', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
       'Punjabi Bagh', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']


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
    print(itr)
    if itr%5==0:
        pool_cpy = Pool.copy()
        station_to_add = max_variance_sampling(Pool,learner_list)
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




random2 = []
random_final = [start]
seed = [i+1 for i in range(50)]
abs_mean = {i:0 for i in range(30)}
start_days = 15
for elem in seed:
    np.random.seed(elem)

## Replace NSIT with Punjabii Bah\gh
    station_train = ['NSIT Dwarka','Delhi Technological University','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
    station_test  = ['Anand Vihar']
    station_pool  = ['North Campus, Delhi - IMD', 'R K Puram',
           'Pusa, Delhi - IMD', 'Aya Nagar, Delhi - IMD', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
           'Punjabi Bagh', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']

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
        day = start_days
  
    for itr in range(31):
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
         #   predicted.append(prediction.mean())
            mae.append(mean_absolute_error(prediction, y_test))
        abs_mean[itr] = np.mean(mae)
    random2.append(deepcopy(abs_mean))
    abs_mean = {}
    
'''
    Change the notation 
    '''

# new_dict = {i:{j:0 for j in range(50)} for i in range(30)}

for i in range(31):
    temp = 0
    for j in random2:
        for k in j:
            if k==i:
                temp+=j[k]
    temp/=len(seed)
    random_final.append(temp)





tmp = np.array(pd.DataFrame(random2).std(axis=0))
err = np.append(np.array([0]),tmp)
import matplotlib
latexify(columns=1)
plt.plot(active_learning,marker = '.',label='Active Sampling')
for j in range(1,31,5):
    plt.axvline(j,ls='--',color='k',lw=0.5)
format_axes(plt.gca())

plt.errorbar([i for i in range(32)],random_final,yerr=err,marker='.',label='Random Sampling')
# plt.title('Estimation at Station $3$ - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='best', shadow=False)
plt.tight_layout()
#plt.savefig('AL Plots/'+station_test[0]+'1.pdf', dpi=300)
