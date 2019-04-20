
# coding: utf-8

# In[2]:



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy
import matplotlib


# In[3]:


df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
station_list = list(df['Station'].unique())


# In[4]:


station = station_list[0]
gt_station  = pd.read_csv('Data/Ground Truth/'+station+'.csv')
predicted = pd.read_csv('Data/Predicted/'+station+'.csv')
gt_station = gt_station.loc[:, ~gt_station.columns.str.contains('^Unnamed')]
predicted = predicted.loc[:,~predicted.columns.str.contains('^Unnamed')]


# In[5]:


predicted


# In[6]:


from scipy.stats import spearmanr


# In[9]:


station_list


# In[15]:


iden = 0
for station in station_list:
    iden+=1
    gt_station  = pd.read_csv('Data/Ground Truth/'+station+'.csv')
    predicted = pd.read_csv('Data/Predicted/'+station+'.csv')
    gt_station = gt_station.loc[:, ~gt_station.columns.str.contains('^Unnamed')]
    predicted = predicted.loc[:,~predicted.columns.str.contains('^Unnamed')]
    y = []
    x = []
    for itr, day in enumerate(range(15,46)):
        y.append(spearmanr(gt_station,predicted[str(day)]))
        x.append(day)
    plt.figure()
    plt.scatter(x,[y[i].correlation for i in range(len(y))])
    plt.xlabel('Days Elapsed')
    plt.ylabel('Spearman Coefficient')
    plt.title(station + str(iden))
    plt.ylim(-0.2,1)
    
    plt.savefig('Data/Correlation Plots/'+station+'.png')


# In[19]:


len(x)


# In[21]:


y[0].correlation


# In[22]:




