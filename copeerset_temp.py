#********co-peer set based on industry and co_code**************************
from APIs import sql_queries as sql
#from scipy import spatial
from sklearn.neighbors import KDTree
from progressbar import ProgressBar
import sys
import pandas as pd
import numpy as np
import copy 
sys.path.append(r'D:\MainRepo')
from Data_Normalization import norm_method_select as norm1 
from Data_Normalization import ind_norm as norm

methods = [norm.winsor,norm.boxcox,norm.minmax,norm.decscaling,norm.zscore,norm.mediannorm,norm.sigmoid,norm.mad,norm.tanh,norm.tansigmoid,norm.meannorm,norm.stdnorm,norm.meanad,norm.scn]
method_name=['winsorized mean','box cox','min max','decimal scaling','Z score','median','sigmoid','Median and Median absolute deviation','Tanh','tansigmoid','mean','standard deviation','mean and mad','statistical column scaling']

def generate_indicator_list(connection):
    indicator = sql.search_table(connection, 'sync')
    cols = []
    for col in indicator.columns:
        if 'year' not in col:
            if col != 'co_code':
                cols.append(col)
    indicator.drop(cols, axis=1, inplace=True)
    temp2 = indicator.transpose()
    temp1 = indicator.drop(['co_code'], axis=1)
    temp = temp1.transpose()
    new_df = pd.DataFrame()
    new_df['yearend'] = temp[0]
    new_df['co_code'] = temp2.ix[0, 0]
    temp3 = pd.DataFrame()
    for i in range(temp.shape[1]):
        if i != 0:
            temp3['yearend'] = temp[i]
            temp3['co_code'] = temp2.ix[0, i]
            new_df = new_df.append(temp3)
    new_df = new_df.reset_index(drop=True)
    indicator = new_df[new_df['yearend'] != 0]
    indicator = indicator.sort_values(['co_code', 'yearend'], ascending=[True, False])
    indicator = indicator.drop_duplicates(subset=['co_code', 'yearend'])
    indicator = indicator.reset_index(drop=True)
    return indicator

def func_closest5_based_on_yearend(connection):
    #finding 5 closest
    print __doc__
    #getting data and merging
    indicator = generate_indicator_list(connection)
    IS_df = sql.search_col(connection, 'IS', ['co_code', 'yearend', 'totalincome'])
    indicator_df = sql.search_col(connection, 'indicator', ['co_code', 'yearend', 'operating_margin_1'])
    df = pd.merge(IS_df, indicator_df, how='inner', on=['co_code', 'yearend'])
    BS_df = sql.search_col(connection, 'BS', ['co_code', 'yearend', 'netblock'])
    df2 = pd.merge(df, BS_df, how='inner', on=['co_code', 'yearend'])
    #merge causing nan values
    indicator_2 = pd.merge(indicator, df2.drop_duplicates(subset=['co_code', 'yearend']),\
                            how='left', on=['co_code', 'yearend'])
    General_df = sql.search_col(connection, 'General', ['co_code', 'industryasperbse'])
    indicator_2 = pd.merge(indicator_2, General_df, how='left', on=['co_code'])
    #normalizing the data 
    x=norm1.one_method(indicator_2[['totalincome', 'operating_margin_1', 'netblock']],\
                        methods[8], 'tanh', -1.1, 0.05, 0.05)
    indicator_2['totalincome'] = x['totalincome'].copy(deep=True)
    indicator_2['operating_margin_1'] = x['operating_margin_1'].copy(deep=True)
    indicator_2['netblock'] = x['netblock'].copy(deep=True)
    groups = indicator_2.groupby('industryasperbse')
    new_dataframe_1 = pd.DataFrame()
    pbar = ProgressBar()   
    for name, group in pbar(groups):
        #looping for industry
        dataframe_1 = pd.DataFrame(group)
        dataframe_1.dropna(axis=0, inplace=True)
        dataframe_1.reset_index(drop=True, inplace=True)
        groups2 = dataframe_1.groupby('yearend')
        new_dataframe_2 = pd.DataFrame()
        for name1, group2 in groups2:
            dataframe_2 = pd.DataFrame(group2)
            dataframe_2.reset_index(drop=True, inplace=True)
            if dataframe_2.empty:
                continue
            else:
                temp_df = dataframe_2.copy(deep=True)
                temp_df.dropna(axis=0, inplace=True)
                temp_df['1st'] = np.nan
                temp_df['2nd'] = np.nan
                temp_df['3rd'] = np.nan
                temp_df['4th'] = np.nan
                temp_df['distance_1'] = np.nan
                temp_df['distance_2'] = np.nan
                temp_df['distance_3'] = np.nan
                temp_df['distance_4'] = np.nan
                for i in range(temp_df.shape[0]):
                    X = np.array(temp_df.ix[:, 2:5])
                    tree = KDTree(X)
                    if temp_df.shape[0] > 4:
                        dist, ind = tree.query(X[i], k=5)
                        temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'co_code']
                        temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'co_code']
                        temp_df.ix[i, '3rd'] = dataframe_2.ix[ind[0][3], 'co_code']
                        temp_df.ix[i, '4th'] = dataframe_2.ix[ind[0][4], 'co_code']
                        temp_df.ix[i,'distance_1'] = dist[0][1]
                        temp_df.ix[i,'distance_2'] = dist[0][2]
                        temp_df.ix[i,'distance_3'] = dist[0][3]
                        temp_df.ix[i,'distance_4'] = dist[0][4]
                    elif temp_df.shape[0] == 4:
                        dist, ind = tree.query(X[i], k=4)
                        temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'co_code']
                        temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'co_code']
                        temp_df.ix[i, '3rd'] = dataframe_2.ix[ind[0][3], 'co_code']
                        temp_df.ix[i,'distance_1'] = dist[0][1]
                        temp_df.ix[i,'distance_2'] = dist[0][2]
                        temp_df.ix[i,'distance_3'] = dist[0][3]
                    elif temp_df.shape[0] == 3:
                        dist, ind = tree.query(X[i], k=3)
                        temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'co_code']
                        temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'co_code']
                        temp_df.ix[i,'distance_1'] = dist[0][1]
                        temp_df.ix[i,'distance_2'] = dist[0][2]                        
                    elif temp_df.shape[0] == 2:
                        dist, ind = tree.query(X[i], k=2)
                        temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'co_code']
                        temp_df.ix[i,'distance_1'] = dist[0][1]                        
                    else:
                        pass
            new_dataframe_2 = new_dataframe_2.append(temp_df)
        new_dataframe_1 = new_dataframe_1.append(new_dataframe_2)
    new_dataframe_1.reset_index(drop=True, inplace=True)
    new_dataframe_1.to_csv(r'D:\MainRepo\Indicators\copeersetpeerset_with_distance.csv',index=False)
    print 'Final Indicator_QTR shape:', new_dataframe_1.shape
    try:
        print "Uploading"
        new_dataframe_1.to_sql('peerset', connection['engine'], if_exists='replace',\
                                chunksize=1000, index=False)
        print "done Uploading"
    except:
        print sys.exc_info()
    del(new_dataframe_1)
    return 0
    
#function to chekc the distance of the two points with the current co_code(group2)
def find_distance(df,code,first,second):
    #convert the dataframe to an array
    df.reset_index(drop=True,inplace=True)
    df1 = df[['1st', '2nd', '3rd', '4th']].copy(deep=True)
    temp_df_array = np.array(df1)
    indices = np.where(temp_df_array == first)
    #print "indices=",indices
    #print "indices[1]=",indices[1]
    #print "len(indices[0])=",len(indices[0])
    dist1 = 0
    for i in range((len(indices[0]))):
        if indices[1][i] == 0:
            dist1 = dist1 + df.ix[indices[0][i],'distance_1']
            #print "df.ix[indices[0][i],'distance_1']=",df.ix[indices[0][i],'distance_1']
            #print "df.ix[indices[0][i],'distance_1']",df.ix[indices[0][i],'distance_1']
        elif indices[1][i] == 1:
            dist1 = dist1 + df.ix[indices[0][i],'distance_2']
            #print "df.ix[indices[0][i],'distance_2']=",df.ix[indices[0][i],'distance_2']
            #print "df.ix[indices[0][i],'distance_2']",df.ix[indices[0][i],'distance_2']
        elif indices[1][i] == 2:
            dist1 = dist1 + df.ix[indices[0][i],'distance_3']
            #print "df.ix[indices[0][i],'distance_3']",df.ix[indices[0][i],'distance_3']
            #print "df.ix[indices[0][i],'distance_3']",df.ix[indices[0][i],'distance_3']
        elif indices[1][i] == 3:
            dist1 = dist1 + df.ix[indices[0][i],'distance_4'] 
            #print "df.ix[indices[0][i],'distance_4']",df.ix[indices[0][i],'distance_4']
            #print "df.ix[indices[0][i],'distance_4']",df.ix[indices[0][i],'distance_4']        
    indices2 = np.where(temp_df_array == second)
    #print "indices2=",indices2
    dist2 = 0
    for i in range((len(indices2[0]))):
        if indices2[1][i] == 0:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_1']
            #print "df.ix[indices2[0][i],'distance_1']=",df.ix[indices2[0][i],'distance_1']
            
        elif indices2[1][i] == 1:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_2']
            #print "df.ix[indices2[0][i],'distance_2']=",df.ix[indices2[0][i],'distance_2']

        elif indices2[1][i] == 2:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_3']
            #print "df.ix[indices2[0][i],'distance_3']=",df.ix[indices2[0][i],'distance_3']

        elif indices2[1][i] == 3:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_4']
            #print "df.ix[indices2[0][i],'distance_4']=",df.ix[indices2[0][i],'distance_4']
    print "dist1 = ",dist1
    print "dist2 = ",dist2
    return dist1,dist2

#function to calculate the distance for a single point(co_code) with the current co_code(group2)
def find_distance1(df,point):
    df.reset_index(drop=True,inplace=True)
    df1 = df[['1st', '2nd', '3rd', '4th']].copy(deep=True)
    temp_df_array = np.array(df1)
    indices = np.where(temp_df_array == point)
    #print "indices_fd1=",indices
    dist1 = 0
    for i in range(len(indices[0])):
        if indices[1][i] == 0:
            dist1 = dist1 + df.ix[indices[0][i],'distance_1']
            #print "df.ix[indices[0][i],'distance_1']=",df.ix[indices[0][i],'distance_1']
  
          #print "df.ix[indices[0][i],'distance_1']",df.ix[indices[0][i],'distance_1']
        elif indices[1][i] == 1:
            dist1 = dist1 + df.ix[indices[0][i],'distance_2']
            #print "df.ix[indices[0][i],'distance_2']=",df.ix[indices[0][i],'distance_2']

            #print "df.ix[indices[0][i],'distance_2']",df.ix[indices[0][i],'distance_2']
        elif indices[1][i] == 2:
            dist1 = dist1 + df.ix[indices[0][i],'distance_3']
            #print "df.ix[indices[0][i],'distance_3']=",df.ix[indices[0][i],'distance_3']

            #print "df.ix[indices[0][i],'distance_3']",df.ix[indices[0][i],'distance_3']
        elif indices[1][i] == 3:
            dist1 = dist1 + df.ix[indices[0][i],'distance_4'] 
            #print "df.ix[indices[0][i],'distance_4']=",df.ix[indices[0][i],'distance_4']

            #print "df.ix[indices[0][i],'distance_4']",df.ix[indices[0][i],'distance_4']    
    #print "only dist1=",dist1
    return dist1

def func_closest5_based_on_cocode(connection):
    dataframe = pd.read_csv(r'D:\MainRepo\Indicators\copeerset\peerset_with_distance.csv')
    groups1 = dataframe.groupby('industryasperbse')   
    new_ind_1 = pd.DataFrame()
    pbar = ProgressBar()   
    #print "Starting",
    for name1, group1 in pbar(groups1):
        dataframe_1 = pd.DataFrame(group1)
        dataframe_1.reset_index(drop=True, inplace=True)
        groups2 = dataframe_1.groupby('co_code')
        new_ind_2 = pd.DataFrame(columns=[['co_code', '1st', '2nd', '3rd', '4th','distance1','distance2','distance3','distance4']])
        for name2, group2 in groups2:
            #print name1
            #print name2
            dataframe_2 = pd.DataFrame(group2)
            dataframe_2.reset_index(drop=True,inplace=True)
            temp_dataframe = dataframe_2[['1st', '2nd', '3rd', '4th']].copy(deep=True)
            if temp_dataframe.isnull().all().all():
                continue
            else:
                #temp_dataframe = dataframe_2[['1st', '2nd', '3rd', '4th']].copy(deep=True)
                var_for_value_count = temp_dataframe.stack().value_counts()
                var_for_index_of_value_counts = list(map(int, var_for_value_count.index))
                #****************************************     
                #check for diatance   
                dict_of_value_count = dict(var_for_value_count)
                from operator import itemgetter
                tuple_of_value_count = sorted(dict_of_value_count.items(), key=itemgetter(1),reverse=True)
                #print "tuple_of_value_count",tuple_of_value_count             
                i = 0    
                #list to store the closest 5 co_codes
                l = []      
                l2 = []
                while i<len(tuple_of_value_count):
                    if i == (len(tuple_of_value_count)-1):
                        l.append(tuple_of_value_count[i][0])
                        dist = find_distance1(dataframe_2, tuple_of_value_count[i][0])
                        l2.append(dist)
                        break
                    elif tuple_of_value_count[i][1] == tuple_of_value_count[i+1][1]:
                        #print "i=",i
                        #print "i+1",i+1
                        #print "tuple_of_value_count[i][0]=",tuple_of_value_count[i][0]
                        #print "tuple_of_value_count[i+1][0]",tuple_of_value_count[i+1][0]
                        dist1, dist2 = find_distance(dataframe_2,name2,int(tuple_of_value_count[i][0]),int(tuple_of_value_count[i+1][0]))
                        #print "dist1=",dist1
                        #print "dist2=",dist2
                        if dist1 < dist2:
                            l.append(tuple_of_value_count[i][0])
                            l2.append(dist1)
                        else:
                            l.append(tuple_of_value_count[i+1][0])
                            l2.append(dist2)
                            #interchange values in original list also
                            temp = tuple_of_value_count[i]
                            tuple_of_value_count[i] = tuple_of_value_count[i+1]
                            tuple_of_value_count[i+1] = temp
                    elif tuple_of_value_count[i][1] > tuple_of_value_count[i+1][1]:
                        l.append(tuple_of_value_count[i][0])  
                        dist = find_distance1(dataframe_2, tuple_of_value_count[i][0])
                        l2.append(dist)
                    i += 1
                    
                #*****************************************                                           
                var_for_index_of_value_counts = copy.deepcopy(l) 
                var_to_store_distance = copy.deepcopy(l2)
                var_for_index_of_value_counts = var_for_index_of_value_counts[0:4]
                var_to_store_distance = var_to_store_distance[0:4]
                if len(var_for_index_of_value_counts) == 3:
                    var_for_index_of_value_counts.append(np.nan)
                    var_to_store_distance.append(np.nan)
                elif len(var_for_index_of_value_counts) == 2:
                    var_for_index_of_value_counts.append(np.nan)
                    var_for_index_of_value_counts.append(np.nan)
                    var_to_store_distance.append(np.nan)
                    var_to_store_distance.append(np.nan)
                elif len(var_for_index_of_value_counts) == 1:
                    var_for_index_of_value_counts.append(np.nan)
                    var_for_index_of_value_counts.append(np.nan)
                    var_for_index_of_value_counts.append(np.nan)
                    var_to_store_distance.append(np.nan)
                    var_to_store_distance.append(np.nan)
                    var_to_store_distance.append(np.nan)
                #print "varA=", var_for_index_of_value_counts
                var_for_index_of_value_counts.append(int(dataframe_2['co_code'].unique()))
                #print "var=", var_for_index_of_value_counts
                #np.roll return array
                var_for_index_of_value_counts = np.roll(var_for_index_of_value_counts, 1)  
                var_to_store_distance = np.array(var_to_store_distance)
                concat_var = np.concatenate((var_for_index_of_value_counts,var_to_store_distance),axis=0)
                x = pd.DataFrame(np.array([concat_var]),\
                                            columns=['co_code', '1st', '2nd', '3rd', '4th','distance1','distance2','distance3','distance4'])
                new_ind_2 = new_ind_2.append(x, ignore_index=True)
        new_ind_1 = new_ind_1.append(new_ind_2)
        #print ".",
        #sys.stdout.flush()
    print "Done!"
    new_ind_1.reset_index(drop=True, inplace=True)
    new_ind_1.to_csv(r'D:\MainRepo\Indicators\copeerset\nearest4.csv', index=False)
    #upload to database
    genreal_dataframe = sql.search_table(connection, 'General')
    general_1 = pd.merge(genreal_dataframe, new_ind_1, how='left', on=['co_code'])
    try:
        print "Uploading "
        general_1.to_sql('general_1', connection['engine'], if_exists='replace',\
                                chunksize=1000, index=False)
    except:
        print sys.exc_info()
    return 0        
        
def main(connection):
    #**main**
    print __doc__
    func_closest5_based_on_yearend(connection)
    func_closest5_based_on_cocode(connection)
    return 0
