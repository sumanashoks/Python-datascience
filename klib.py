#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip insatll klib')


# In[3]:


get_ipython().system('pip install klib')


# In[4]:


pip install --upgrade tables


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('/Users/apple/Desktop/python/googleplaystore.csv',usecols=['Category','Rating','Reviews','Size','Installs','Type','Price'])


# In[3]:


df


# In[4]:


import klib


# In[5]:


klib.cat_plot(df)


# In[6]:


klib.corr_mat(df)


# In[7]:


klib.dist_plot(df)


# In[8]:


klib.missingval_plot(df)


# In[11]:


klib.clean


# In[12]:


klib.data_cleaning(df)


# In[13]:


klib.clean_column_names(df)


# In[14]:


klib.convert_datatypes(df) 


# In[15]:


klib.drop_missing(df)


# In[16]:


klib.mv_col_handling(df)


# In[17]:


klib.pool_duplicate_subsets(df)


# In[18]:


df


# In[19]:


klib.cat_plot(df, top=4, bottom=4)


# In[20]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df,explorative=True)
profile.to_file('output.html')


# In[1]:


import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('/Users/apple/Desktop/python/googleplaystore.csv', usecols=['Category','Rating','Reviews','Size','Installs','Type','Price'])

profile = ProfileReport(df, title="Profiling Report")
profile.to_file('output.html')


# In[2]:


df


# In[23]:


df = pd.read_csv('/Users/apple/Desktop/python/googleplaystore.csv', usecols=['Category','Rating','Reviews','Size','Installs','Type','Price'])


# # PANDAS PROFILING
# # DTALE
# # KLIB
# # pygwalker

# In[1]:


pip install pygwalker


# In[1]:


import pandas as pd
import pygwalker as pyg


# In[2]:


df=pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[3]:


pyg.walk(df)


# In[4]:


df.info()


# In[4]:


vis_spec = r"""{"config":[{"config":{"defaultAggregated":true,"geoms":["auto"],"coordSystem":"generic","limit":-1},"encodings":{"dimensions":[{"dragId":"gw_pJ4S","fid":"gender","name":"gender","basename":"gender","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_hCuh","fid":"smoker","name":"smoker","basename":"smoker","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_QTQI","fid":"day","name":"day","basename":"day","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_5HpC","fid":"time","name":"time","basename":"time","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_ZR-P","fid":"size","name":"size","basename":"size","semanticType":"quantitative","analyticType":"dimension"},{"dragId":"gw_mea_key_fid","fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"}],"measures":[{"dragId":"gw_bDVS","fid":"total_bill","name":"total_bill","basename":"total_bill","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_A-Jw","fid":"tip","name":"tip","basename":"tip","analyticType":"measure","semanticType":"quantitative","aggName":"sum"},{"dragId":"gw_count_fid","fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"dragId":"gw_mea_val_fid","fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"dragId":"gw_Qgy7","fid":"total_bill","name":"total_bill","basename":"total_bill","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"columns":[{"dragId":"gw_YTEk","fid":"smoker","name":"smoker","basename":"smoker","semanticType":"nominal","analyticType":"dimension"},{"dragId":"gw_pr3d","fid":"day","name":"day","basename":"day","semanticType":"nominal","analyticType":"dimension"}],"color":[{"dragId":"gw_k6lg","fid":"time","name":"time","basename":"time","semanticType":"nominal","analyticType":"dimension"}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[{"dragId":"gw_XRZ7","fid":"gender","name":"gender","basename":"gender","semanticType":"nominal","analyticType":"dimension"}],"filters":[{"dragId":"gw_cRIr","fid":"smoker","name":"smoker","basename":"smoker","semanticType":"nominal","analyticType":"dimension","rule":null}],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"fixed","width":541,"height":300},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false}},"visId":"gw_USpM","name":"Chart 1"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["smoker","day","time","gender"],"measures":[{"field":"total_bill","agg":"sum","asFieldKey":"total_bill_sum"}]}]}]}],"version":"0.4.4"}"""
pyg.walk(df, spec=vis_spec)


# In[ ]:




