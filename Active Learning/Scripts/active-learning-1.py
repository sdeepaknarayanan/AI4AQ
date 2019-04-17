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
    station_to_add = station
    train_to_add_1 = df_main[df_main['Station']==station_to_add][day:day+1]
    train_to_add_2 = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_train])
    x_train = np.append(x_train,train_to_add_2[['Latitude','Longitude','Date']],axis = 0)
    x_train =  np.append(x_train, train_to_add_1[['Latitude','Longitude','Date']],axis=0)
    y_train = np.append(y_train, train_to_add_1['PM2.5'])
    y_train = np.append(y_train, train_to_add_2['PM2.5'])
    # Pool set remove
    station_train.append(station_to_add)
    pool_df = pool_df[pool_df['Station']!=station_to_add]
    station_pool.remove(station_to_add)
    pool_to_add = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
    pool_df = pd.concat([pool_df,pool_to_add,pool_to_add]).drop_duplicates(keep=False)
    return x_train, y_train, pool_df

def daily_update(df_main, x_train, y_train, pool_df,day):
    ## Else add the remaining data at the needed locations
    train_to_add_2 = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_train])
    x_train = np.append(x_train,train_to_add_2[['Latitude','Longitude','Date']],axis = 0)
    y_train = np.append(y_train, train_to_add_2['PM2.5'])
    pool_to_add = pd.concat([df_main.groupby('Station').get_group(i)[day:day+1] for i in station_pool])
    pool_df = pd.concat([pool_df,pool_to_add,pool_to_add]).drop_duplicates(keep=False)
    return x_train, y_train, pool_df

station_train = ['Punjabi Bagh','Aya Nagar, Delhi - IMD','Burari Crossing, Delhi - IMD','CRRI Mathura Road, Delhi - IMD']
station_test  = ['Delhi Technological University']
station_pool  = ['North Campus, Delhi - IMD', 'Anand Vihar',
       'Pusa, Delhi - IMD', 'R K Puram', 'Shadipur', 'Siri Fort', 'Income Tax Office', 'Lodhi Road, Delhi - IMD', 'Mandir Marg',
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