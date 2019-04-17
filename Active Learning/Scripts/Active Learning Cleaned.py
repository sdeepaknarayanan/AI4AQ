
# coding: utf-8

# ## Active Learning to estimate PM2.5 at CPCB Stations
# * Added Time, Latitude and Longitude as parameters

# <p> Standard imports are done here.... As we can see, we are importing the Regressor function from the ModAL wrapper. This module is extremely useful as we'll see below </p>

# 'Anand Vihar' - Station 1, 'Aya Nagar, Delhi - IMD',
#        'Burari Crossing, Delhi - IMD', 'CRRI Mathura Road, Delhi - IMD',
#        'Delhi Technological University' - Station 2,
#        'IGI Airport Terminal-3, Delhi - IMD', 'IHBAS',
#        'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
#        'NSIT Dwarka', 'North Campus, Delhi - IMD', 'Punjabi Bagh',
#        'Pusa, Delhi - IMD', 'R K Puram' - Station 3, 'Shadipur', 'Siri Fort'
#        

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy

# In[8]:


df = pd.read_csv('/home/deepak/Projects and Coursework/CS399/Datasets/Updated_Delhi_Scaled.csv')


# In[9]:


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()


# In[21]:


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
    # Add station to training data
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
    ## Else add the remaining data at the needed locations
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


# In[22]:


initial_number_of_days = 15


# In[23]:


station_train = ['Punjabi Bagh','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['R K Puram']
station_pool  = ['North Campus, Delhi - IMD', 'Anand Vihar',
       'Pusa, Delhi - IMD', 'Delhi Technological University', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
       'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']

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


# In[24]:


#start_days = 15
day = 15
daylist = df['Date'].unique()
#mae = {i:0 for i in range(5)}
mae_=np.zeros(5)
day = day
active_learning = np.zeros(31)
active_learning[0] = start

for itr, day in enumerate(range(15,46)):
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
    active_learning[itr]=np.mean(mae)


# In[73]:


random2 = []
#mae = {i:0 for i in range(5)}
#mae_=np.zeros(5)
random_final = [start]
seed = [i+1 for i in range(50)]
abs_mean = {i:0 for i in range(30)}
start_days = 15
for elem in seed:
    station_train = ['Punjabi Bagh','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
    station_test  = ['R K Puram']
    station_pool  = ['North Campus, Delhi - IMD', 'Anand Vihar',
           'Pusa, Delhi - IMD', 'Delhi Technological University', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
           'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']

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
        np.random.seed(elem)
        day = start_days
  
    for itr in range(30):
        if itr%5==0:
            station_to_add = np.random.choice(station_pool)
            x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
        else:
            x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
        day+=1
        prediction = []
        mae = []
        predicted = []
        for i in learner_list:
            learner_list[i].fit(x_train, y_train)
            prediction = (learner_list[i].predict(x_test))
            predicted.append(prediction.mean())
            mae.append(mean_absolute_error(prediction, y_test))
        abs_mean[itr] = np.mean(mae)
    random2.append(deepcopy(abs_mean))
#     print(abs_mean)
    abs_mean = {}
    
'''
    Change the notation 
    '''
new_dict = {i:{j:0 for j in range(50)} for i in range(30)}
for i in range(30):
    temp = 0
    for j in random2:
        for k in j:
            if k==i:
                temp+=j[k]
    temp/=len(seed)
    random_final.append(temp)


# In[74]:

'''
from copy import deepcopy
random_f1 = deepcopy(random_final)
random1 = deepcopy(random)
del random
del random_final

'''
# In[75]:

'''
n2 = []
random1 = []
mae = {i:0 for i in range(5)}
mae_=np.zeros(5)
random_final = [start]
seed = [i+1 for i in range(50)]
abs_mean = {i:0 for i in range(30)}
start_days = 15
for elem in seed:
    station_train = ['Punjabi Bagh','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
    station_test  = ['R K Puram']
    station_pool  = ['North Campus, Delhi - IMD', 'Anand Vihar',
           'Pusa, Delhi - IMD', 'Delhi Technological University', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
           'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']

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
        np.random.seed(elem)
        day = start_days
  
    for itr in range(30):
        if itr%5==0:
            station_to_add = np.random.choice(station_pool)
            x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
        else:
            x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
        day+=1
        prediction = []
        mae = []
        predicted = []
        for i in learner_list:
            learner_list[i].fit(x_train, y_train)
            prediction = (learner_list[i].predict(x_test))
            predicted.append(prediction.mean())
            mae.append(mean_absolute_error(prediction, y_test))
        abs_mean[itr] = np.mean(mae)
        if itr==0:
            n2.append(abs_mean[itr])
    random1.append(deepcopy(abs_mean))
#     n2.append()
    abs_mean = {}
for i in range(30):
    temp = 0
    for j in random1:
        for k in j:
            if k==i:
                temp+=j[k]
    temp/=len(seed)
    random_final.append(temp)
'''

# In[78]:


#random[0] ,random1[0]


# In[71]:


#random_final,random_f1


# ## Below - Data initialisation

# station_train = ['Punjabi Bagh','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
# station_test  = ['R K Puram']
# station_pool  = ['North Campus, Delhi - IMD', 'Anand Vihar',
#        'Pusa, Delhi - IMD', 'Delhi Technological University', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
#        'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']
# 
# train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
# train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
# x_train = train[['Latitude','Longitude','Date']]
# y_train = train['PM2.5']
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# 
# Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
# Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
# x_pool = Pool[['Latitude','Longitude','Date']]
# y_pool = Pool['PM2.5']
# x_pool = np.array(x_pool)
# y_pool = np.array(y_pool)
# 
# Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
# Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
# x_test = Test[['Latitude','Longitude','Date']]
# y_test = Test['PM2.5']
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# learner_list = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
# for i in learner_list:
#     learner_list[i].fit(x_train,y_train)
# mae = np.zeros(5)
# for i in range(5):
#     mae[i] = mean_absolute_error(learner_list[i+1].predict(x_test),y_test)
# print('The initial MAE Mean is ',mae.mean())
# start = mae.mean()
# 

# for itr in range(30):
#         if itr%5==0:
#             station_to_add = np.random.choice(station_pool)
#             x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
#         else:
#             x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
#         day+=1
#         prediction = []
#         mae = []
#         for i in learner_list:
#             learner_list[i].fit(x_train, y_train)
#             prediction = (learner_list[i].predict(x_test))
#             mae.append(mean_absolute_error(prediction, y_test))
# #             print(mean_absolute_error(prediction,y_test))
#         abs_mean[itr] = np.mean(mae)
#     random.append(abs_mean)

# In[35]:


active_learning, random_final


# In[37]:


#err


# In[44]:


tmp = np.array(pd.DataFrame(random2).std(axis=0))
err = np.append(np.array([0]),tmp)
from math import sqrt
import matplotlib
latexify(columns=1)
plt.plot(active_learning,marker = '.',label='Active Sampling',alpha=0.4)
for j in range(1,31,5):
    plt.axvline(j,ls='--',color='k',lw=0.5)
format_axes(plt.gca())

plt.errorbar([i for i in range(31)],random_final,yerr=err,marker='.',label='Random Sampling',alpha=0.8)
# plt.title('Estimation at Station $3$ - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='best', shadow=False)
plt.tight_layout()
plt.savefig('Station3_12.pdf', dpi=300)


# In[28]:

'''
active_learning


# In[18]:


pd.DataFrame(random)


# In[45]:


get_ipython().run_line_magic('pinfo', 'plt.errorbar')


# In[50]:


tmp = np.append(np.array([0]),err)


# In[38]:


err = np
err = np.array(pd.DataFrame(random).std(axis=0))
err.shape


# In[52]:


tmp


# for test_station in list_station:
#     
#     new_station_list = list_station.copy()
#     new_station_list.remove(test_station)
#     np.random.shuffle(new_station_list)
#     
#     station_train = new_station_list[12:]
#     station_test  = [test_station]
#     station_pool  = new_station_list[:12]
#     
#     #print(station_train, station_test, station_pool)
#     
#     train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
#     train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
#     x_train = train[['Latitude','Longitude','Date']]
#     y_train = train['PM2.5']
#     x_train = np.array(x_train)
#     y_train = np.array(y_train)
# 
#     Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
#     Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
#     x_pool = Pool[['Latitude','Longitude','Date']]
#     y_pool = Pool['PM2.5']
#     x_pool = np.array(x_pool)
#     y_pool = np.array(y_pool)
# 
#     Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
#     Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
#     x_test = Test[['Latitude','Longitude','Date']]
#     y_test = Test['PM2.5']
#     x_test = np.array(x_test)
#     y_test = np.array(y_test)
#     learner_list = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
#     for i in learner_list:
#         learner_list[i].fit(x_train,y_train)
#     mae = np.zeros(5)
#     for i in range(5):
#         mae[i] = mean_absolute_error(learner_list[i+1].predict(x_test),y_test)
#     start = mae.mean()
#     
#     start_days = 10
#     day = start_days
#     daylist = df['Date'].unique()
#     mae = {i:0 for i in range(5)}
#     mae_=np.zeros(5)
#     day = day
#     active_learning = [start]
# 
#     for itr in range(30):
#         if itr%5==0:
#             station_to_add = max_variance_sampling(Pool,learner_list)
#             x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
#         else:
#             x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
#         day+=1
#         prediction = []
#         mae = []
#         for i in learner_list:
#             learner_list[i].fit(x_train, y_train)
#             prediction = (learner_list[i].predict(x_test))
#             mae.append(mean_absolute_error(prediction, y_test))
#         active_learning.append(np.mean(mae))
#     
#     random = []
#     mae = {i:0 for i in range(5)}
#     mae_=np.zeros(5)
#     random_final = [start]
#     seed = [i for i in range(30)]
#     abs_mean = {i:[] for i in range(30)}
#     start_days = 10
#     for elem in seed:
#         station_train = new_station_list[12:]
#         station_test  = [test_station]
#         station_pool  = new_station_list[:12]
#         train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
#         train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
#         x_train = train[['Latitude','Longitude','Date']]
#         y_train = train['PM2.5']
#         x_train = np.array(x_train)
#         y_train = np.array(y_train)
# 
#         Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
#         Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
#         x_pool = Pool[['Latitude','Longitude','Date']]
#         y_pool = Pool['PM2.5']
#         x_pool = np.array(x_pool)
#         y_pool = np.array(y_pool)
# 
#         Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
#         Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
#         x_test = Test[['Latitude','Longitude','Date']]
#         y_test = Test['PM2.5']
#         x_test = np.array(x_test)
#         y_test = np.array(y_test)
#         np.random.seed(elem)
#         day = start_days
# 
#         for itr in range(30):
#             if itr%5==0:
#                 # Add every 5 days
#                 # Train set add 
#                 station_to_add = np.random.choice(station_pool)
#                 x_train, y_train, Pool = update_pool_train(station_to_add, df, x_train, y_train, Pool,day)
#             else:
#                 x_train, y_train, Pool = daily_update(df,x_train,y_train,Pool,day)
#             day+=1
#             prediction = []
#             mae = []
#             for i in learner_list:
#                 learner_list[i].fit(x_train, y_train)
#                 prediction = (learner_list[i].predict(x_test))
#                 mae.append(mean_absolute_error(prediction, y_test))
#             abs_mean[itr] = np.mean(mae)
#         random.append(abs_mean)
#     for i in range(30):
#         temp = 0
#         for j in random:
#             for k in j:
#                 if k==i:
#                     temp+=j[k]
#         temp/=len(seed)
#         random_final.append(temp)
#     plt.figure()
#     plt.plot([i for i in range(31)],active_learning,marker = '.',label='With Active Learning')
#     plt.plot([i for i in range(31)],random_final,marker='.',label='Random Choice')
#     plt.title('Estimation at '+test_station+' - MAE vs Days')
#     plt.xlabel('Number of Days Elapsed')
#     plt.ylabel('Mean Absolute Error')
#     plt.legend(loc='best', shadow=True) 
'''