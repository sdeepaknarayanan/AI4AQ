
# coding: utf-8

# ## Active Learning to estimate PM2.5 at CPCB Stations
# * Added Time, Latitude and Longitude as parameters

# <p> Standard imports are done here.... As we can see, we are importing the Regressor function from the ModAL wrapper. This module is extremely useful as we'll see below </p>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from modAL.models import ActiveLearner, CommitteeRegressor
#from modAL.disagreement import max_std_sampling
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import random as rd


# In[2]:


df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')


# In[3]:


df.head()


# In[4]:


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()


# In[5]:


len((df['Station'].unique()))


# In[6]:


### Nice workaround for removing a specific column
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[7]:


station_train = ['Anand Vihar','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['Delhi Technological University']
station_pool  = ['North Campus, Delhi - IMD', 'Punjabi Bagh',
       'Pusa, Delhi - IMD', 'R K Puram', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
       'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']
len(station_train),len(station_pool)


# ### Here is how we'll be doing Active Learning
# 1. Suppose the initial number of stations we have in our train set is 4.
# 2. Let us have 7 stations in our pool set.
# 3. Let us also have the remaining 6 stations in the test set. 
# 
# ### We need to predict at these 6 stations by quering from our pool data available.
# 1. Let's assume we start off with data for the say, first 15 days for our 4 stations.
#     * The dataset that we're using has data for 50 days in total.
# 2. We compute initial error on the test stations that we have chosen, for these 15 days. Now, we'll add a new station from the pool set to our training set, from day 16 onwards. 
#     * We add the data only from data 16 onwards. This is because it is equivalent to installing a new sensor at the location of the new station.
# 3. Now, after adding this new station, we add the data for a few more days from this station too, say 5 days. Then we estimate the error in the PM2.5 for our test stations using this data. Note that the test has changed now, we have total 20 days worth of data. We simultaneously sample a random station from the pool data we have and obtain corresponding results for that station too as described and we iterate.
# 
# ### Queries related to Querying from the pool
# 1. What is my pool set and how do I measure the standard deviation?
#     * Do we fix our pool or do we add data to it?
#          * What I mean is we have 15 days worth data for all the stations. Then do we look at the entire 
#            data that's present in the pool (all 50 days) or only for those 15 days for the stations in the pool?
#       * Also, can we also query only for day 16 for all the stations from the pool?
#       * Also, we'll be Querying 'a' station from the pool. So, is the mean a better metric assuming I am taking all the stations into account from the pool?
# 
# ### What did I do here for querying?
# * I took all the values of the stations, correspondingly for a given latitude
# * I remove all its instances from the pool data (For first query - 15 of them, second query 16 and so on..)
#     
# ### Querying
# * Take all the points, compute the standard deviation for all the points. 
#     * For each station, take the mean of the standard deviation, for the 10 days
#     * Then, take that station that has the maximum standard deviation

# ### Initial stations have been chosen for train data. The number of days we're initially using is, 15.
# ### I am creating the Train, Pool and Test Datasets here.

# In[8]:


learner_list = {i:KNeighborsRegressor(n_neighbors=i+1, weights='distance') for i in range(5)}
learners = {i:KNeighborsRegressor(n_neighbors=i+1, weights='distance') for i in range(5)}


# In[9]:


initial_number_of_days = 15


# In[10]:


station_train = ['Delhi Technological University','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['Anand Vihar']
station_pool  = ['North Campus, Delhi - IMD', 'Punjabi Bagh',
       'Pusa, Delhi - IMD', 'R K Puram', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
       'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']
len(station_train),len(station_pool)
#initial_number_of_days = 15

train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
#train = train.drop(['Station'],axis = 1)
train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
x_train = train[['Latitude','Longitude','Date']]
y_train = train['PM2.5']
x_train = np.array(x_train)
y_train = np.array(y_train)

Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
#Pool = Pool.drop(['Station'],axis = 1)
Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
x_pool = Pool[['Latitude','Longitude','Date']]
y_pool = Pool['PM2.5']
x_pool = np.array(x_pool)
y_pool = np.array(y_pool)

Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
#Test = Test.drop(['Station'],axis = 1)
Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
x_test = Test[['Latitude','Longitude','Date']]
y_test = Test['PM2.5']
x_test = np.array(x_test)
y_test = np.array(y_test)


# learner_list = [ActiveLearner(estimator=KNeighborsRegressor(n_neighbors=i,weights = 'distance'),
#                 X_training=x_train,y_training = y_train)
#                 for i in range(1,6) ]
# mae = np.zeros(5)
# for i in range(5):
#     mae[i] = mean_absolute_error(ActiveLearner.predict(learner_list[i],x_test),y_test)
# tmp = mae.mean()
# print("The Initial MAE for k varying from 1 to 5 is,",mae) 
# print("\nMean of the initial MAE is ",tmp)

# In[12]:


for i in learner_list:
    learner_list[i].fit(x_train,y_train)


# In[22]:


def pool_strategy(Pool,current_day):
    t = {}
    for i in range(len(learner_list)):
        t[i+1] = learner_list[i].predict(Pool[['Latitude','Longitude','Date']]) 
#     print(pd.DataFrame(t))
    df_ = pd.DataFrame(t).std(axis = 1)
#     print(df_)
    i=0
    rs = 0
    tmp = []
    while(i<len(Pool)):
        if i!=0 and i%current_day==current_day-1:
            rs/=current_day
            tmp.append(rs)
            rs = 0
        else:
            rs+=df_[i]
        i+=1
    #print(tmp,len(tmp))
    pool_remove = Pool.iloc[[np.array(tmp).argmax()*current_day+1]]   
    train_add = (df.groupby('Station').get_group(Pool.iloc[[np.array(tmp).argmax()*current_day+1]].iloc[0]['Station'])[current_day:current_day+1])
    return pool_remove, train_add, pd.DataFrame(t)


# In[24]:


_,_,c = pool_strategy(Pool, current_day)


# In[30]:


station_pool


# In[35]:


df_temp = pd.DataFrame()
for i in station_pool:
    _ = Pool.groupby('Station').get_group(i)[['Latitude','Longitude','PM2.5']]
    t = {}
    for i in range(len(learner_list)):
        t[i+1] = learner_list[i].predict(_) 

    print(_)
    break


# In[37]:


pd.DataFrame(t)


# In[ ]:


new_df = pd.DataFrame()
for station in station_pool:
    new_df['Station']


# In[14]:


learners = {i:KNeighborsRegressor(n_neighbors=i, weights='distance') for i in range(1,6)}
for i in learners:
    learners[i].fit(x_train,y_train)
mae = np.zeros(5)
for i in range(5):
    mae[i] = mean_absolute_error(learners[i+1].predict(x_test),y_test)
print('The initial MAE Mean is ',mae.mean())
start = mae.mean()


# In[15]:


current_day = 15


# In[16]:


#committee = CommitteeRegressor(learner_list=learner_list,query_strategy=max_std_sampling)
daylist = df['Date'].unique()
mae = {i:0 for i in range(5)}
mae_=np.zeros(5)

## Adding the first station to the train --- from the pool 
#query_idx, query_instance = committee.query(x_pool)
#committee.teach(query_instance,y_pool[query_idx])
## Current Day maintain
#temp = pd.DataFrame(x_pool)
#temp_ = temp.groupby(0).get_group(x_pool[query_idx][0][0])
#temp = pd.concat([temp,temp_,temp_]).drop_duplicates(keep=False)
#to_add = df.groupby('Latitude').get_group(x_pool[query_idx][0][0]).iloc[[current_day]]
day = current_day
active_learning = [start]
for itr in range(30):
    #print(itr)
    if itr%5==0:
        pool_,train_ = pool_strategy(Pool,day)
        #break
        temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_train])
        x_train  = np.append(x_train,train_[['Latitude','Longitude','Date']],axis = 0)
        x_train = np.append(x_train,temp[['Latitude','Longitude','Date']],axis = 0)
        print(train_[['Station']].iloc[0][0])
        station_train.append(train_[['Station']].iloc[0][0])
        y_train =  np.append(y_train,train_[['PM2.5']])
        y_train = np.append(y_train,temp[['PM2.5']])
        temp = Pool.groupby('Station').get_group(pool_[['Station']].iloc[0][0])
        station_pool.remove(pool_[['Station']].iloc[0][0])
        Pool = pd.concat([Pool,temp,temp]).drop_duplicates(keep=False)
        new = pd.DataFrame()
        for i in station_pool:
            tmp = Pool.groupby('Station').get_group(i)
            tmp = tmp.sort_values(by=['Date'])
            new = pd.concat([new,tmp])
        Pool = new
    else:
        temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_train])
        t1 = pd.DataFrame(x_train)
        #print(t1)
        x_train = np.append(x_train,temp[['Latitude','Longitude','Date']],axis = 0)
        y_train = np.append(y_train,temp[['PM2.5']])
        t1 = pd.DataFrame(x_train)
        temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
        temp= temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
        Pool = pd.concat([Pool,temp],ignore_index=True)
        Pool = Pool.sort_values(by='Station')
    day+=1
    for i in learner_list:
        learner_list[i].fit(x_train,y_train)
    mae = np.zeros(5)
    for i in range(5):
        mae[i] = mean_absolute_error(learner_list[i].predict(x_test),y_test)
    active_learning.append(mae.mean())
    print(mae.mean())
    
    #print(Pool)
    #time.sleep(5)


# In[16]:


new


# In[17]:


plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.')
plt.title('Estimation at Anand Vihar with Active Learning - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,123,'NSIT Dwarka - Day 1')
plt.text(16.5,120,'IGI Airport T3 - Day 6')
plt.text(16.5,117,'Pusa, Delhi - Day 11')
plt.text(16.5,114,'Punjabi Bagh - Day 16')
plt.text(16.5,111,'North Campus, Delhi - Day 21')
plt.text(16.5,108,'RK Puram - Day 26')
#plt.savefig('AL_Mathura1.pdf')


# In[18]:


Pool


# In[19]:


station_train = ['Anand Vihar','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['Delhi Technological University']
station_pool  = ['North Campus, Delhi - IMD', 'Punjabi Bagh',
       'Pusa, Delhi - IMD', 'R K Puram', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
       'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']
len(station_train),len(station_pool)
#initial_number_of_days = 15

train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
#train = train.drop(['Station'],axis = 1)
train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
x_train = train[['Latitude','Longitude','Date']]
y_train = train['PM2.5']
x_train = np.array(x_train)
y_train = np.array(y_train)

Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
#Pool = Pool.drop(['Station'],axis = 1)
Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
x_pool = Pool[['Latitude','Longitude','Date']]
y_pool = Pool['PM2.5']
x_pool = np.array(x_pool)
y_pool = np.array(y_pool)

Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
#Test = Test.drop(['Station'],axis = 1)
Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
x_test = Test[['Latitude','Longitude','Date']]
y_test = Test['PM2.5']
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[21]:


#committee = CommitteeRegressor(learner_list=learner_list,query_strategy=max_std_sampling)
daylist = df['Date'].unique()
mae = {i:0 for i in range(5)}
mae_=np.zeros(5)

## Adding the first station to the train --- from the pool 
#query_idx, query_instance = committee.query(x_pool)
#committee.teach(query_instance,y_pool[query_idx])
## Current Day maintain
#temp = pd.DataFrame(x_pool)
#temp_ = temp.groupby(0).get_group(x_pool[query_idx][0][0])
#temp = pd.concat([temp,temp_,temp_]).drop_duplicates(keep=False)
#to_add = df.groupby('Latitude').get_group(x_pool[query_idx][0][0]).iloc[[current_day]]
seed = [i for i in range(100)]
abs_mean = {i:[] for i in range(30)}
for elem in seed:
    station_train = ['Delhi Technological University','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
    station_test  = ['Anand Vihar']
    station_pool  = ['North Campus, Delhi - IMD', 'Punjabi Bagh',
           'Pusa, Delhi - IMD', 'R K Puram', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
           'NSIT Dwarka', 'IHBAS','IGI Airport Terminal-3, Delhi - IMD']
    len(station_train),len(station_pool)
    #initial_number_of_days = 15

    train = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_train])
    #train = train.drop(['Station'],axis = 1)
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    x_train = train[['Latitude','Longitude','Date']]
    y_train = train['PM2.5']
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    Pool = pd.concat([df.groupby('Station').get_group(i)[:initial_number_of_days]for i in station_pool])
    #Pool = Pool.drop(['Station'],axis = 1)
    Pool = Pool.loc[:, ~Pool.columns.str.contains('^Unnamed')]
    x_pool = Pool[['Latitude','Longitude','Date']]
    y_pool = Pool['PM2.5']
    x_pool = np.array(x_pool)
    y_pool = np.array(y_pool)

    Test = pd.concat([df.groupby('Station').get_group(i) for i in station_test])
    #Test = Test.drop(['Station'],axis = 1)
    Test = Test.loc[:,~Test.columns.str.contains('^Unnamed')]
    x_test = Test[['Latitude','Longitude','Date']]
    y_test = Test['PM2.5']
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    rd.seed(elem)
    day = current_day
    for itr in range(30):
        #print(itr)
        if itr%5==0:
            sid = rd.randint(0,len(station_pool)-1)
            curr_stn = station_pool[sid]
            #print(curr_stn)
            train_ = df.groupby('Station').get_group(curr_stn)[day:day+1]
            Pool = Pool[Pool.Station!=curr_stn]
            temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_train])
            x_train  = np.append(x_train,train_[['Latitude','Longitude','Date']],axis = 0)
            x_train = np.append(x_train,temp[['Latitude','Longitude','Date']],axis = 0)
            station_train.append(train_[['Station']].iloc[0][0])
            y_train =  np.append(y_train,train_[['PM2.5']])
            y_train = np.append(y_train,temp[['PM2.5']])
            #temp = Pool.groupby('Station').get_group(pool_[['Station']].iloc[0][0])
            station_pool.remove(curr_stn)
            #Pool = pd.concat([Pool,temp,temp]).drop_duplicates(keep=False)
        else:
            temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_train])
            t1 = pd.DataFrame(x_train)
            #print(t1)
            x_train = np.append(x_train,temp[['Latitude','Longitude','Date']],axis = 0)
            y_train = np.append(y_train,temp[['PM2.5']])
            t1 = pd.DataFrame(x_train)
            temp = pd.concat([df.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
            temp= temp.loc[:, ~temp.columns.str.contains('^Unnamed')]
            Pool = pd.concat([Pool,temp],ignore_index=True)
            Pool = Pool.sort_values(by='Station')
        day+=1
        for i in learners:
            learners[i].fit(x_train,y_train)
        mae = np.zeros(5)
        for i in range(5):
            mae[i] = mean_absolute_error(learners[i+1].predict(x_test),y_test)
        #print(mae.mean())
        abs_mean[itr].append(mae.mean())
random_final = [start]
for i in abs_mean:
    random_final.append(np.mean(abs_mean[i]))


# In[22]:


random_final[0] = start


# In[23]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'NSIT Dwarka - Day 1')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('1.pdf')


# In[24]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'IGI Airport T3 - Day 6')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('2.pdf')


# In[25]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'Pusa, Delhi - Day 11')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('3.pdf')


# In[26]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'Punjabi Bagh - Day 16')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('4.pdf')


# In[27]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'North Campus, Delhi - Day 21')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('5.pdf')


# In[28]:


#markers_on = [12, 17, 18, 19,20,21]
plt.plot([i for i in range(31)],active_learning,color = 'red',marker = '.',label='With Active Learning')
plt.plot([i for i in range(0,31)],random_final,c='black',marker='.',label='Random Choice')
plt.title('Estimation at Anand Vihar - MAE vs Days')
plt.xlabel('Number of Days Elapsed')
plt.ylabel('Mean Absolute Error')
plt.text(16.5,108,'RK Puram - Day 26')
#plt.savefig('AL_Mathura1.pdf')
#plt.plot([i for i in range(0,31)],random_final,c='black',marker = '.')
#plt.plot(markevery=markers_on,marker = '|',c='blue')
#plt.plot(mae.keys(),mae.values(),c='black',marker = '.'

#plt.title('Estimation at Anand Vihar - Random Baseline - MAE vs Days')
plt.legend(loc='best', shadow=True)
#plt.savefig('6.pdf')


# In[29]:


plt.text(16.5,108,'IGI Airport T3 - Day 6')
plt.text(16.5,108,'Pusa, Delhi - Day 11')
plt.text(16.5,108,'Punjabi Bagh - Day 16')
plt.text(16.5,108,'North Campus, Delhi - Day 21')
plt.text(16.5,108,'RK Puram - Day 26')

