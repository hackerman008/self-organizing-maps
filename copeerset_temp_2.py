"""co-peerset version-3 with correlation values"""

import sys
sys.path.append(r'D:\MainRepo')
from APIs import sql_queries as sql
from APIs import common_func 
#from scipy import spatial
from sklearn.neighbors import KDTree
from scipy import stats
from progressbar import ProgressBar
import pandas as pd
import numpy as np
import copy 
from itertools import combinations
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
    
def find_correlation(unique_cocode_values, df, indicator, df_with_price_returns):
    
    """finding the correlation within each pair using the values from the dataframe"""
#****************************************************************************************    
    #make sure the dataframe that is passed is properly indexed
    #df = indicator.copy(deep=True) 
    
    #condition to check if the datarframe is empty 
    #todo    
    
    indicator['year_round'] = indicator['yearend']//100

    """testing"""    
    first = indicator.groupby('industryasperbse').get_group('2/3 Wheelers')
    df = first.groupby('year_round').get_group(2016.0)
    df = df.reset_index(drop=True)
    
    #column_names = df.columns.values[np.in1d(df.columns.values,['co_code', 'industryasperbse'],assume_unique=True,invert=True)]    
    co_codes_list = df.ix[:,'co_code'].unique()
    unique_comb = [list(comb) for comb in combinations(co_codes_list, 2)]    
    #print "unique_combination for co_code=",unique_comb           
    
    df['year_round'] = df['yearend']//100    
    df['year_round'] = df['year_round'].astype(int)     
    start_year = df['year_round'].unique() 
    end_year = start_year - 5
    df_temp = indicator[indicator['year_round'].isin(range(start_year, end_year,-1))]
#***************************************************************************************

#***************************************************************************************
    """for price returns calculate the start date and end date.
    list to store the start date to enddate values"""
    #extract the date from the df intermediate dataframe
    start_year = df['yearend'].unique() #float
    
    temp = start_year/100 #float with month in decial place
    start_year_int = int(temp)
    start_year_month_value = temp%start_year_int
    end_year_int = start_year_int - 5
    range_of_dates_int = range(start_year_int, end_year_int, -1)
    range_of_dates_with_month = [(date+start_year_month_value)*100 for date in range_of_dates_int]
    first = range_of_dates_with_month[0]
    second = range_of_dates_with_month[1]
    df_temp2 = df_with_price_returns[(df_with_price_returns['yearend'] <= first[0]) & (df_with_price_returns['yearend'] >= second[0])]
        
    

#***************************************************************************************
    
    """testing correlation calculation for sales_growth"""
#    temp1 = df_temp[df_temp['co_code']==6]
#    temp1 = temp1.reset_index(drop=True)
#    temp2 = df_temp[df_temp['co_code']==17]
#    temp2 = temp2.reset_index(drop=True)    
#    a = min(temp1.shape[0], temp2.shape[0])
#    temp1 = temp1.ix[~(temp1.index >=a)]
#    temp2 = temp2.ix[~(temp1.index >=a)]
#    return_corr, check_val  = common_func.calculate_correlation(list(temp1.ix[:, 'sales_growth_streak']), list(temp2.ix[:, 'sales_growth_streak']))
#    if check_val == False:
#        corr_sales_growth_streak = temp1[['sales_growth_streak']].corrwith(temp2[['sales_growth_streak']])        
#    elif check_val == True:
#        corr_sales_growth_streak = return_corr

#    a=temp1[['sales_growth_streak']].corrwith(temp2[['sales_growth_streak']])
    """testing correlation calculation for price_returns"""
#    temp1 = df_temp2[df_temp2['co_code']==6]
#    temp1 = temp1.reset_index(drop=True)
#    temp2 = df_temp2[df_temp2['co_code']==9]
#    temp2 = temp2.reset_index(drop=True)
#    a = min(temp1.shape[0], temp2.shape[0])
#    temp1 = temp1.ix[~(temp1.index >=a)]
#    temp2 = temp2.ix[~(temp1.index >=a)]
#    a=temp1[['price_returns']].corrwith(temp2[['price_returns']])


    list_for_correlation_values_sales_growth = []
    list_for_correlation_values_price_returns = []
    for i in unique_comb:
        #i = [151.0, 1334.0]
        print "i[0]=",i[0]
        print "i[1]=",i[1]
        """for correlation calculation using sales_growth_streak"""
        temp1 = df_temp[df_temp['co_code']==i[0]]
        temp1 = temp1.reset_index(drop=True)
        temp2 = df_temp[df_temp['co_code']==i[1]]
        temp2 = temp2.reset_index(drop=True)
        minimum_row_value = min(temp1.shape[0], temp2.shape[0])
        #remove the extra rows in the dataframe with extra rows
        temp1 = temp1.ix[~(temp1.index >= minimum_row_value)]
        temp2 = temp2.ix[~(temp2.index >= minimum_row_value)]
        #find the correlation
        return_corr, check_val  = common_func.calculate_correlation(list(temp1.ix[:, 'sales_growth_streak']), list(temp2.ix[:, 'sales_growth_streak']))
        if check_val == False:
            corr_sales_growth_streak = temp1[['sales_growth_streak']].corrwith(temp2[['sales_growth_streak']])       
            corr_sales_growth_streak = corr_sales_growth_streak[0]
        elif check_val == True:
            corr_sales_growth_streak = return_corr
        print "corr_sales_growth_streak=",corr_sales_growth_streak
        list_for_correlation_values_sales_growth.append((i,corr_sales_growth_streak)) 
        
        """for correlation calculation using price_returns"""
        temp3 = df_temp2[df_temp2['co_code']==i[0]]
        temp3 = temp3.reset_index(drop=True)
        temp4 = df_temp2[df_temp2['co_code']==i[1]]
        temp4 = temp4.reset_index(drop=True)
        minimum_row_value2 = min(temp3.shape[0], temp4.shape[0])
        temp3 = temp3.ix[~(temp3.index >= minimum_row_value2)]
        temp4 = temp4.ix[~(temp4.index >= minimum_row_value2)]
        #check if the one of the two arrays has all values same or different
        return_corr, check_val  = common_func.calculate_correlation(list(temp3.ix[:, 'price_returns']), list(temp4.ix[:, 'price_returns']))
        if check_val == False:
            corr_price_returns = temp3[['price_returns']].corrwith(temp4[['price_returns']])
            corr_price_returns = corr_price_returns[0]
        elif check_val == True:
            corr_price_returns = return_corr  
        print "corr_price_returns=", corr_price_returns
        list_for_correlation_values_price_returns.append((i,corr_price_returns)) 

 
    """testing with respect to one co_code"""  
#    temp_list_of_correlation_values = []
#    indices=[i for i,val in enumerate(unique_comb) if val[0]==151.0 or val[1]==151.0]
#    temp_list_of_correlation_values = [list_for_correlation_values_sales_growth[val][1] for val in indices]    


    for i in df.index: #or for i in co_codes_list:
        """testing for index=0 ,for the first co_code"""
            
        temp_list_of_correlation_values = []
        indices=[j for j,val in enumerate(unique_comb) if val[0]==151.0 or val[1]==151.0]
        temp_list_of_correlation_values = [list_for_correlation_values_sales_growth[val][1] for val in indices]    
        temp_list_of_correlation_values2 = [list_for_correlation_values_price_returns[val][1] for val in indices]    
        #i=0
        #print "for i=",i
        #insert the value=1 at the proper index location
        temp_list_of_correlation_values.insert(i, 1)
        temp_list_of_correlation_values2.insert(i, 1)
        #convert the list into a series
        se = pd.Series(temp_list_of_correlation_values)
        se2 = pd.Series(temp_list_of_correlation_values2)
        #append the series with correlation values to the dataframe
        df['col_for_correlation_values_sales_growth_streak'] = se.values
        df['col_for_correlation_values_price_returns'] = se2.values        
        column_names = list(df.columns.values[np.in1d(df.columns.values,['co_code', 'yearend', 'industryasperbse', 'companyname','year_round'],assume_unique=True,invert=True)])    
    
        #******************************************************************    
        """distance calculation using kdtree"""
        X = df[column_names].values
        tree = KDTree(X)
        dist, ind = tree.query(X[i], k=1)
        #print "indi=",ind
        #print "dist=",dist
        
        #append the thing to the dataframe
        #todo

        #df.drop(labels=['col_for_correlation_values_sales_growth_streak', 'col_for_correlation_values_price_returns'], axis=1, inplace=True)
    
        #*****************************************************************
        """distance calculation using pairwise distances"""
        #    temp_df_1 = df[df['co_code'] == df.ix[i,'co_code']]
        #    X = temp_df_1[column_names].values
        #    temp_df_2 = df[df['co_code'] != df.ix[i,'co_code']]        
        #    Y = temp_df_2[column_names].values   
        #    #calculate the distances of the current co_code with respect to other co_codes
        #    from sklearn.metrics.pairwise import pairwise_distances 
        #    distances = pairwise_distances(X, Y)
        #******************************************************************
        #or 
        
    return 0
    
def find_distance():
    distance =0
    
    return distance

def func_closest5_based_on_yearend(connection):
    """function to find the nearest neighbor of each stock
       indicators:- sales_growth, operating margin, fixedasset by totalassets over 5 yrs, 
       fixed asset turnover, totalincome, price_returns
    """
    
    #indicators - sales_growth, operating margin, fixedasset by totalassets over 5 yrs, fixed asset turnover, totalincome, price_returns
    indicator = generate_indicator_list(connection) 
    df_fixed_asset_turnover = sql.search_col(connection, 'indicator', ['co_code', 'yearend', 'fixed_asset_turnover','operating_margin_1', \
                                                                        'sales_growth_streak'])
    df_totalincome = sql.search_col(connection, 'IS', ['co_code', 'yearend', 'totalincome'])
    df1 = pd.merge(df_fixed_asset_turnover, df_totalincome, how='left', on=['co_code', 'yearend'])
    
    df_totalassets = sql.search_col(connection, 'BS', ['co_code', 'yearend', 'totalassets'])
    df2 = pd.merge(df1, df_totalassets, how='inner', on=['co_code', 'yearend'])

    df_fixedassets = sql.search_col(connection, 'Ratio', ['co_code', 'yearend', 'fixedassets'])
    df3 = pd.merge(df2, df_fixedassets, how='inner', on=['co_code', 'yearend'])

    df_price_returns = sql.search_col(connection, 'indicator_monthly', ['co_code', 'yearend', 'price_returns'])
    df4 = pd.merge(df3, df_price_returns, how='inner', on=['co_code', 'yearend'])

    indicator_industryasperbse = sql.search_col(connection, 'General', ['co_code', 'industryasperbse', 'companyname'])
    df5 = pd.merge(df4, indicator_industryasperbse, how='left', on=['co_code'])

    indicator = pd.merge(indicator, df5, how='left', on=['co_code', 'yearend'])
   
    indicator['fixedassets_by_totalassets'] = indicator['fixedassets'].div(indicator['totalassets'])
    indicator['fixedassets_by_totalassets_over_5yrs'] = indicator.groupby('co_code',group_keys=False).apply(lambda x:x['fixedassets_by_totalassets'].rolling(window=5,min_periods=1).mean().shift(-4))
    for i in range(4,0,-1):
        indicator['temp_'+str(i)] = indicator.groupby('co_code',group_keys=False).apply(lambda x:x['fixedassets_by_totalassets'].rolling(window=i,min_periods=1).mean().shift(-(i-1)))
        indicator['fixedassets_by_totalassets_over_5yrs'] = np.where(indicator['fixedassets_by_totalassets_over_5yrs'].isnull(), indicator['temp_'+str(i)], indicator['fixedassets_by_totalassets_over_5yrs'])
        indicator.drop(labels=['temp_'+str(i)],axis=1,inplace=True)    
    indicator.drop(labels=['fixedassets_by_totalassets','fixedassets','totalassets'],axis=1,inplace=True)    
   
    indicator['operating_margin_over_5_yrs'] = indicator.groupby('co_code',group_keys=False).apply(lambda x:x['operating_margin_1'].rolling(window=5,min_periods=1).mean().shift(-4))
    for i in range(4,0,-1):
        indicator['temp_'+str(i)] = indicator.groupby('co_code',group_keys=False).apply(lambda x:x['operating_margin_1'].rolling(window=i,min_periods=1).mean().shift(-(i-1)))
        indicator['operating_margin_over_5_yrs'] = np.where(indicator['operating_margin_over_5_yrs'].isnull(), indicator['temp_'+str(i)], indicator['operating_margin_over_5_yrs'])
        indicator.drop(labels=['temp_'+str(i)],axis=1,inplace=True)    
    indicator.drop(labels=['operating_margin_1'],axis=1,inplace=True)    


    #normalizing the data 
    x=norm1.one_method(indicator[['fixed_asset_turnover', 'sales_growth_streak', 'totalincome','price_returns','fixedassets_by_totalassets_over_5yrs','operating_margin_over_5_yrs']],\
                        methods[8], 'tanh', -1.1, 0.05, 0.05)
    indicator['fixed_asset_turnover'] = x['fixed_asset_turnover'].copy(deep=True)
    indicator['sales_growth_streak'] = x['sales_growth_streak'].copy(deep=True)
    indicator['totalincome'] = x['totalincome'].copy(deep=True)
    indicator['price_returns'] = x['price_returns'].copy(deep=True)
    indicator['fixedassets_by_totalassets_over_5yrs'] = x['fixedassets_by_totalassets_over_5yrs'].copy(deep=True)
    indicator['operating_margin_over_5_yrs'] = x['operating_margin_over_5_yrs'].copy(deep=True)
    #***********************************************
    
    df_market_cap = sql.search_col(connection,'Price', ['co_code', 'yearend', 'marketcap'])
    df_price_returns =sql.search_col(connection, 'indicator_monthly', ['co_code', 'yearend', 'price_returns'])
    df_with_price_returns = pd.merge(df_price_returns, df_market_cap, how='inner', on=['co_code','yearend'])    

    first = indicator.groupby('industryasperbse').get_group('2/3 Wheelers')
    df = first.groupby('yearend').get_group(201603.0)
    df = df.reset_index(drop=True)
    
    #column_names = df.columns.values[np.in1d(df.columns.values,['co_code', 'industryasperbse'],assume_unique=True,invert=True)]    
    co_codes_list = df.ix[:,'co_code'].unique()    
    find_correlation(1, indicator, df_with_price_returns)

#**************************todo************************************************    
    groups = indicator.groupby('industryasperbse')
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
            print "for year=",name1
            dataframe_2 = pd.DataFrame(group2)
            dataframe_2.reset_index(drop=True, inplace=True)
            #todo
            find_correlation(1, dataframe_2, indicator, df_with_price_returns)
            
#            new_dataframe_2 = new_dataframe_2.append(temp_df)
#        new_dataframe_1 = new_dataframe_1.append(new_dataframe_2)
#    new_dataframe_1.reset_index(drop=True, inplace=True)

#***************************************************************    

    return 0




if __name__ == "__main__":
    
    func_closest5_based_on_yearend(connection)
    return 0
