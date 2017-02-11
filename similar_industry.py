import sys
sys.path.append(r'D:\MainRepo')
from APIs import sql_queries as sql
#from scipy import spatial
from sklearn.neighbors import KDTree
from progressbar import ProgressBar
import pandas as pd
import numpy as np
import copy 
from Data_Normalization import norm_method_select as norm1 
from Data_Normalization import ind_norm as norm
from operator import itemgetter
import time
import logging
import psycopg2
from sqlalchemy import create_engine

methods = [norm.winsor,norm.boxcox,norm.minmax,norm.decscaling,norm.zscore,norm.mediannorm,norm.sigmoid,norm.mad,norm.tanh,norm.tansigmoid,norm.meannorm,norm.stdnorm,norm.meanad,norm.scn]
method_name=['winsorized mean','box cox','min max','decimal scaling','Z score','median','sigmoid','Median and Median absolute deviation','Tanh','tansigmoid','mean','standard deviation','mean and mad','statistical column scaling']

def generate_indicator_list(connection,_aggregation_attr):
    if _aggregation_attr == "industryasperbse":
        indicator = sql.search_table(connection,'ts_iapbse_sync')
    elif _aggregation_attr == "sector":
        indicator = sql.search_table(connection,'ts_sector_sync')
    elif _aggregation_attr == "industry":
        indicator = sql.search_table(connection,'ts_industry_sync')
        
    cols = []
    for col in indicator.columns:
        if 'year' not in col:
            if(col!=_aggregation_attr):
                cols.append(col)
    indicator.drop(cols,axis=1,inplace=True)
    #print(indicator.head())
    #return
    temp2 = indicator.transpose()  
    temp1 = indicator.drop([_aggregation_attr],axis=1)
    temp = temp1.transpose()
    new_df = pd.DataFrame()
    new_df['yearend'] = temp[0]
    new_df[_aggregation_attr] = temp2.ix[0,0]
    temp3 = pd.DataFrame()
 
    for i in range(temp.shape[1]):
        if(i!=0):
            temp3['yearend'] = (temp[i])
            
            temp3[_aggregation_attr] = (temp2.ix[0,i])
            new_df = new_df.append(temp3)
            
    new_df = new_df.reset_index(drop=True)
 
    indicator = new_df[new_df['yearend']!=0]
    indicator = indicator.sort_values([_aggregation_attr,'yearend'],ascending=[True,False])
    indicator = indicator.drop_duplicates(subset = [_aggregation_attr, 'yearend'])
    indicator = indicator.reset_index(drop=True) 
    indicator['yearend'] = indicator['yearend'].astype(float)
    #indicator['yearendm'] = indicator['yearend']*100 + 3
    #print(indicator)
    return indicator
    
#function to chekc the distance of the two points with the current industryasperbse(group2)
def find_distance(df, code, first,second):
    #convert the dataframe to an array
    df.reset_index(drop=True, inplace=True)    
    df1 = df[['1st', '2nd', '3rd', '4th']].copy(deep=True)
    #print 'df1',df1
    temp_df_array = np.array(df1)
    indices = np.where(temp_df_array == first)
    print "indices=", indices
    #print "indices[1]=",indices[1]
    #print "len(indices[0])=",len(indices[0])
    dist1 = 0
    for i in range((len(indices[0]))):
        if indices[1][i] == 0:
            dist1 = dist1 + df.ix[indices[0][i], 'distance_1']
        elif indices[1][i] == 1:
            dist1 = dist1 + df.ix[indices[0][i], 'distance_2']
        elif indices[1][i] == 2:
            dist1 = dist1 + df.ix[indices[0][i], 'distance_3']
        elif indices[1][i] == 3:
            dist1 = dist1 + df.ix[indices[0][i], 'distance_4'] 
    indices2 = np.where(temp_df_array == second)
    print "indices2=",indices2
    dist2 = 0
    for i in range((len(indices2[0]))):
        if indices2[1][i] == 0:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_1']
        elif indices2[1][i] == 1:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_2']
        elif indices2[1][i] == 2:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_3']
        elif indices2[1][i] == 3:
            dist2 = dist2 + df.ix[indices2[0][i],'distance_4']
    #print "dist1 = ",dist1
    #print "dist2 = ",dist2
    return dist1,dist2

#function to calculate the distance for a single point(industryasperbse) with the current industryasperbse(group2)
def find_distance1(df,point):
    df.reset_index(drop=True,inplace=True)
    df1 = df[['1st', '2nd', '3rd', '4th']].copy(deep=True)
    #print "df1=",df1
    temp_df_array = np.array(df1)
    indices = np.where(temp_df_array == point)
    print "indices_fd1=",indices
    dist1 = 0
    for i in range(len(indices[0])):
        if indices[1][i] == 0:
            dist1 = dist1 + df.ix[indices[0][i],'distance_1']
        elif indices[1][i] == 1:
            dist1 = dist1 + df.ix[indices[0][i],'distance_2']
        elif indices[1][i] == 2:
            dist1 = dist1 + df.ix[indices[0][i],'distance_3']
        elif indices[1][i] == 3:
            dist1 = dist1 + df.ix[indices[0][i],'distance_4'] 
    #print "only dist1=",dist1
    return dist1
    
def find_nearest_neighbor(connection, _aggregation_attr):
    
    #_aggregation_attr = "industryasperbse"
    var = 'ts_iapbse_'
    indicator = generate_indicator_list(connection,_aggregation_attr)

    df = sql.search_col(connection,var+'BS',[_aggregation_attr,'yearend','netblock'])
    df1 = sql.search_col(connection,var+'IS',[_aggregation_attr,'yearend','totalincome'])
    df2 = pd.merge(df,df1,how='inner',on=['yearend',_aggregation_attr])
    
    df3 = sql.search_col(connection,var+'indicator',[_aggregation_attr,'yearend','i_operating_margin_1'])
    df4 = pd.merge(df2,df3,how='left',on=['yearend',_aggregation_attr])
    indicator_2 = pd.merge(indicator, df4, how='left', on=['yearend',_aggregation_attr])  
    
    #normalizing the data 
    x=norm1.one_method(indicator_2[['totalincome', 'i_operating_margin_1', 'netblock']], methods[8], 'tanh', -1.1, 0.05, 0.05)
    indicator_2['totalincome'] = x['totalincome'].copy(deep=True)
    indicator_2['i_operating_margin_1'] = x['i_operating_margin_1'].copy(deep=True)
    indicator_2['netblock'] = x['netblock'].copy(deep=True)

    #temp_df2 = indicator_2.groupby('yearend').get_group(2016.0) #return's dataframe
    groups2 = indicator_2.groupby('yearend')
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
                    temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'industryasperbse']
                    temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'industryasperbse']
                    temp_df.ix[i, '3rd'] = dataframe_2.ix[ind[0][3], 'industryasperbse']
                    temp_df.ix[i, '4th'] = dataframe_2.ix[ind[0][4], 'industryasperbse']
                    temp_df.ix[i,'distance_1'] = dist[0][1]
                    temp_df.ix[i,'distance_2'] = dist[0][2]
                    temp_df.ix[i,'distance_3'] = dist[0][3]
                    temp_df.ix[i,'distance_4'] = dist[0][4]
                elif temp_df.shape[0] == 4:
                    dist, ind = tree.query(X[i], k=4)
                    temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'industryasperbse']
                    temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'industryasperbse']
                    temp_df.ix[i, '3rd'] = dataframe_2.ix[ind[0][3], 'industryasperbse']
                    temp_df.ix[i,'distance_1'] = dist[0][1]
                    temp_df.ix[i,'distance_2'] = dist[0][2]
                    temp_df.ix[i,'distance_3'] = dist[0][3]
                elif temp_df.shape[0] == 3:
                    dist, ind = tree.query(X[i], k=3)
                    temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'industryasperbse']
                    temp_df.ix[i, '2nd'] = dataframe_2.ix[ind[0][2], 'industryasperbse']
                    temp_df.ix[i,'distance_1'] = dist[0][1]
                    temp_df.ix[i,'distance_2'] = dist[0][2]                        
                elif temp_df.shape[0] == 2:
                    dist, ind = tree.query(X[i], k=2)
                    temp_df.ix[i, '1st'] = dataframe_2.ix[ind[0][1], 'industryasperbse']
                    temp_df.ix[i,'distance_1'] = dist[0][1]                        
                else:
                    pass
        new_dataframe_2 = new_dataframe_2.append(temp_df) 
        
    groups2 = new_dataframe_2.groupby('industryasperbse')
    new_ind_2 = pd.DataFrame(columns=[['industryasperbse', '1st', '2nd', '3rd', '4th','distance1','distance2','distance3','distance4']])
    for name2, group2 in groups2:
        print '*********industryasperbse=', name2
        dataframe_2 = pd.DataFrame(group2)
        dataframe_2.reset_index(drop=True,inplace=True)
        temp_dataframe = dataframe_2[['1st', '2nd', '3rd', '4th']].copy(deep=True)
        if temp_dataframe.isnull().all().all():
            continue
        else:
            #temp_dataframe = dataframe_2[['1st', '2nd', '3rd', '4th']].copy(deep=True)
            var_for_value_count = temp_dataframe.stack().value_counts()
            #print "\nvar_for_value_count=",var_for_value_count                 
            var_for_index_of_value_counts = list(var_for_value_count.index)
            #print "\nvar_for_index_of_value_counts=",var_for_index_of_value_counts                
            #check for diatance   
            dict_of_value_count = dict(var_for_value_count)
            #print "\ndict_of_value_count=",dict_of_value_count
            tuple_of_value_count = sorted(dict_of_value_count.items(), key=itemgetter(1),reverse=True)
            print "\ntuple_of_value_count",tuple_of_value_count             
            i = 0    
            #list to store the closest 5 industryasperbses
            l = []      
            l2 = []
            while i<len(tuple_of_value_count):
                #print "\nfor industryasperbse inside tuple list=",tuple_of_value_count[i][0]
                if i == (len(tuple_of_value_count)-1):
                    l.append(tuple_of_value_count[i][0])
                    dist = find_distance1(dataframe_2, tuple_of_value_count[i][0])
                    l2.append(dist)
                    break
                elif tuple_of_value_count[i][1] == tuple_of_value_count[i+1][1]:
                    dist1, dist2 = find_distance(dataframe_2, name2, tuple_of_value_count[i][0], tuple_of_value_count[i+1][0])
                    if dist1 < dist2:
                        l.append(tuple_of_value_count[i][0])
                        l2.append(dist1)
                    else:
                        l.append(tuple_of_value_count[i+1][0])
                        l2.append(dist2)
                        #swap values 
                        temp = tuple_of_value_count[i]
                        tuple_of_value_count[i] = tuple_of_value_count[i+1]
                        tuple_of_value_count[i+1] = temp
                elif tuple_of_value_count[i][1] > tuple_of_value_count[i+1][1]:
                    l.append(tuple_of_value_count[i][0])  
                    dist = find_distance1(dataframe_2, tuple_of_value_count[i][0])
                    l2.append(dist)
                i += 1
                                           
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
            temp_unique_industry_name_for_current_dataframe = str(dataframe_2['industryasperbse'].unique()[0])
            var_for_index_of_value_counts.append(temp_unique_industry_name_for_current_dataframe)
            print "var=", var_for_index_of_value_counts
            #np.roll return array
            var_for_index_of_value_counts = np.roll(var_for_index_of_value_counts, 1)  
            var_to_store_distance = np.array(var_to_store_distance)
            concat_var = np.concatenate((var_for_index_of_value_counts,var_to_store_distance),axis=0)
            x = pd.DataFrame(np.array([concat_var]),\
                                        columns=['industryasperbse', '1st', '2nd', '3rd', '4th','distance1','distance2','distance3','distance4'])
            new_ind_2 = new_ind_2.append(x, ignore_index=True)        
        
    return new_ind_2

def connect(db):
   
    #defining our connection string    
    conn_string = "host='192.168.0.5' port = '5433' dbname='"+db['name']+"' user='postgres' password='"+db['password'] +"'"
    
    #establishing connection    
    #print conn_string
    try:
        conn = psycopg2.connect(conn_string)
        conn.autocommit = True
        engine = create_engine('postgresql://postgres:harileela@192.168.0.5:5433/FundaDB')
        print "connected"
    except:
        print "Oops!" + str(sys.exc_info()) + "Occured .(Connection object)"
        exit
        
            
    #creating a cursor object 
    try:
        cursor = conn.cursor()
        connection = {'conn':conn,'cursor':cursor,'engine':engine}
        #        cursor.execute("SET CLIENT_ENCODING TO 'UTF8';")
    except:
        print "Oops!" + str(sys.exc_info()) + "Occured .(Connection :Cursor Object)"
        return 0
    return connection
    
#==============================================================================
# Setting up a logger to log data into a .txt file
#==============================================================================
    
def log_setup(file_name):
    try:
        logging.basicConfig(level=logging.INFO,filename=file_name+'.log')
        logger = logging.getLogger()
    except:
        print str(sys.exc_info())
    
    return logger
    

if __name__ == "__main__":
    
    logger = log_setup('test'+time.strftime('%d-%m-%Y_%H-%M-%S_%p'))
     #printing log status
    logger.info('Main : Session Started')
    db={'name':'FundaDB','password':'harileela'}
    #connecting to the df    
    connection = connect(db)
    
    _aggregation_attr = 'industryasperbse'
    dataframe_with_nearest_neighbor_for_each_industry = find_nearest_neighbor(connection, _aggregation_attr)