#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('/Users/apple/Downloads/pandascleaning.csv')


# In[3]:


df


# In[8]:


import dtale

dtale.show(df)


# In[16]:


#D-TALE
#Charts Code Export
# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

chart_data = pd.concat([
	df['pulse'],
	df['calories'],
], axis=1)
chart_data = chart_data.sort_values(['pulse'])
chart_data = chart_data.rename(columns={'pulse': 'x'})
chart_data = chart_data.dropna()

import plotly.graph_objs as go

charts = []
charts.append(go.Bar(
	x=chart_data['x'],
	y=chart_data['calories']
))
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h', 'y': -0.3},
    'title': {'text': 'calories by pulse'},
    'xaxis': {'tickformat': '0:g', 'title': {'text': 'pulse'}},
    'yaxis': {'title': {'text': 'calories'}, 'type': 'linear'}
}))

# If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
#
# from plotly.offline import iplot, init_notebook_mode
#
# init_notebook_mode(connected=True)
# for chart in charts:
#     chart.pop('id', None) # for some reason iplot does not like 'id'
# iplot(figure)
# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

chart_data = pd.concat([
	df['pulse'],
	df['calories'],
], axis=1)
chart_data = chart_data.sort_values(['pulse'])
chart_data = chart_data.rename(columns={'pulse': 'x'})
chart_data = chart_data.dropna()

import plotly.graph_objs as go

charts = []
charts.append(go.Bar(
	x=chart_data['x'],
	y=chart_data['calories']
))
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h', 'y': -0.3},
    'title': {'text': 'calories by pulse'},
    'xaxis': {'tickformat': '0:g', 'title': {'text': 'pulse'}},
    'yaxis': {'title': {'text': 'calories'}, 'type': 'linear'}
}))

# If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
#
# from plotly.offline import iplot, init_notebook_mode
#
# init_notebook_mode(connected=True)
# for chart in charts:
#     chart.pop('id', None) # for some reason iplot does not like 'id'
# iplot(figure)



from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)
for chart in charts:
    chart.pop('id', None) # for some reason iplot does not like 'id'
iplot(figure)


# In[ ]:





# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(subset=['Type'],inplace=True)


# In[7]:


df.isnull().sum()


# In[15]:


#D-TALE
#Charts Code Export
# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

chart_data = pd.concat([
	df['Rating'],
	df['Price'],
], axis=1)
chart_data = chart_data.sort_values(['Rating'])
chart_data = chart_data.rename(columns={'Rating': 'x'})
chart_data = chart_data.dropna()

import plotly.graph_objs as go

charts = []
charts.append(go.Bar(
	x=chart_data['x'],
	y=chart_data['Price']
))
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h', 'y': -0.3},
    'title': {'text': 'Price by Rating'},
    'xaxis': {'title': {'text': 'Rating'}},
    'yaxis': {'title': {'text': 'Price'}, 'type': 'linear'}
}))

# If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
#
# from plotly.offline import iplot, init_notebook_mode
#
# init_notebook_mode(connected=True)
# for chart in charts:
#     chart.pop('id', None) # for some reason iplot does not like 'id'
# iplot(figure)
# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

chart_data = pd.concat([
	df['Rating'],
	df['Price'],
], axis=1)
chart_data = chart_data.sort_values(['Rating'])
chart_data = chart_data.rename(columns={'Rating': 'x'})
chart_data = chart_data.dropna()

import plotly.graph_objs as go

charts = []
charts.append(go.Bar(
	x=chart_data['x'],
	y=chart_data['Price']
))
figure = go.Figure(data=charts, layout=go.Layout({
    'barmode': 'group',
    'legend': {'orientation': 'h', 'y': -0.3},
    'title': {'text': 'Price by Rating'},
    'xaxis': {'title': {'text': 'Rating'}},
    'yaxis': {'title': {'text': 'Price'}, 'type': 'linear'}
}))

# If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
#
# from plotly.offline import iplot, init_notebook_mode
#
# init_notebook_mode(connected=True)
# for chart in charts:
#     chart.pop('id', None) # for some reason iplot does not like 'id'
# iplot(figure)


# In[ ]:


from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)
for chart in charts:
    chart.pop('id', None) # for some reason iplot does not like 'id'
iplot(figure)


# In[6]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df,explorative=True)
profile.to_file('output.html')


# In[10]:


pip install --upgrade pandas-profiling


# In[7]:


pip uninstall pandas-profiling
pip install pandas-profiling


# In[ ]:


get_ipython().system('pip uninstall pandas-profiling')


# In[ ]:


Y


# In[ ]:


pip install pandas-profiling


# # SWEETVIZ
# 

# # AUTO VIZ
# # PYGWALKER
# 
# 

# In[1]:


import pandas as pd
import pygwalker as pyg


# In[ ]:





# In[4]:


df = pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')
walker = pyg.walk(df)


# In[5]:


df


# In[6]:


pip install pygwalker --upgrade



# In[ ]:




