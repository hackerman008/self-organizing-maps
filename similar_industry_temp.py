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

def find_nearest_neighbor(connection, _aggregation_attr):
    
    _aggregation_attr = "industryasperbse"
    var = 'ts_iapbse_'
    indicator = generate_indicator_list(connection,_aggregation_attr)

    df_indicator = sql.search_col(connection,var+'indicator',[_aggregation_attr, 'yearend', 'i_revenue_growth_1', 'i_operating_margin_1', 'i_ROIC_1'])
    df_BS = sql.search_col(connection,var+'BS',[_aggregation_attr,'yearend','totalassets'])
    df2 = pd.merge(df_indicator, df_BS,how='left',on=['yearend',_aggregation_attr])
    
    
    df_BS__netblock = sql.search_col(connection,var+'BS',[_aggregation_attr,'yearend','netblock'])
    df3 = pd.merge(df2, df_BS__netblock,how='left',on=['yearend',_aggregation_attr])
    
    #retriev employee cost per from company














    
  