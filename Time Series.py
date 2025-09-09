#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("/Users/apple/Desktop/python/Time Series/sales.csv")


# In[3]:


df


# In[4]:


df.rename(columns={'Perrin Freres monthly champagne sales millions ?64-?72': 'sales'}, inplace=True)


# In[5]:


df


# In[6]:


df.drop(105,inplace=True,axis=0)


# In[7]:


df


# In[8]:


df['sales'].unique()


# In[9]:


df['sales'].value_counts(normalize=True)*100


# In[10]:


df.describe()


# In[11]:


df[df['sales']==13916]


# In[12]:


sns.violinplot(data=df)


# In[42]:


sns.boxplot(data=df)


# In[13]:


df.set_index('Month',inplace=True)


# In[14]:


df


# In[15]:


df


# In[16]:


df.tail(n=4)


# In[19]:


df.drop('Perrin Freres monthly champagne sales millions ?64-?72',inplace=True,axis=0)


# In[20]:


df.tail(n=4)


# In[21]:


df.info()


# In[22]:


df.describe()


# In[24]:


df.dtypes


# In[25]:


df.plot()


# In[27]:


import plotly.express as px

fig = px.line(df, x=df.index, y='sales', title='Sales Over Time')
fig.show()


# # AS WE SEE THE ABOVE GRAPH IT'S SHOWING UPTREND AND DOWNTRED
# 
# # IT IS CALLED NON-STATIONARY 
# 
# # A STATIONARY WILL BE HORIZONTAL / SIDEWAYS IT WILL BE EASY CALCULATE FOR MOVING AVERAGE 
# 
# # So by seeing the graph I can tell it's NON-STATIONARY but the machine also should tell us. so we adfuller test.

# In[28]:


from statsmodels.tsa.stattools import adfuller


# In[30]:


test_result=adfuller(df['sales'])


# In[31]:


#HYPOTHESIS TEST:
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    
    result=adfuller(sales)
    
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[32]:


adfuller_test(df['sales'])


# # If the P-Value is below than 0.5 Then is stationary 

# # Now we should make Stationary

# In[34]:


df['Seasonal First Difference']=df['sales']-df['sales'].shift(12)


# In[35]:


df


# In[36]:


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# # Now have a look that In P-Value at last it's e-11 then it's lesser than 0.5
# 
# # lags used is 0 means it's 'D' 
# # if it is 0 then consider it as 1.
# 
# # now D is 1

# In[43]:


df['Seasonal First Difference'].plot()


# In[44]:


import plotly.graph_objects as go

# Assuming 'Seasonal First Difference' is a column in your DataFrame
fig = go.Figure(go.Scatter(x=df.index, y=df['Seasonal First Difference'], mode='lines', name='Seasonal First Difference'))
fig.update_layout(title='Seasonal First Difference Plot', xaxis_title='Index', yaxis_title='Seasonal First Difference')
fig.show()


# # Now our data is stationary

# In[45]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[47]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['sales'])
plt.show()


# In[49]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# #  Autocorrelation is Q
# # partial autocorrelation is p

# In[52]:


# For non-seasonal data
#p=1, d=1, q=0 or 1


from statsmodels.tsa.arima_model import ARIMA


# In[54]:


from statsmodels.tsa.arima.model import ARIMA

# Create ARIMA model
model = ARIMA(df['sales'], order=(1, 1, 1))

# Fit the model
model_fit = model.fit()


# In[55]:


model_fit.summary()


# In[57]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['sales','forecast']].plot(figsize=(12,8))


# In[58]:


import plotly.graph_objects as go

# Create a Plotly figure
fig = go.Figure()

# Add 'sales' data to the figure
fig.add_trace(go.Scatter(x=df.index, y=df['sales'], mode='lines', name='Actual Sales'))

# Add 'forecast' data to the figure
fig.add_trace(go.Scatter(x=df.index, y=df['forecast'], mode='lines', name='Forecast'))

# Update layout
fig.update_layout(title='Actual Sales vs Forecast', xaxis_title='Index', yaxis_title='Sales')

# Show plot
fig.show()


# # SARIMA

# In[59]:


import statsmodels.api as sm


# In[60]:


model=sm.tsa.statespace.SARIMAX(df['sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[61]:


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['sales','forecast']].plot(figsize=(12,8))


# In[62]:


import plotly.graph_objects as go

# Create traces
trace_actual_sales = go.Scatter(x=df.index, y=df['sales'], mode='lines', name='Actual Sales')
trace_forecast = go.Scatter(x=df.index, y=df['forecast'], mode='lines', name='Forecast')

# Create plotly figure
fig = go.Figure([trace_actual_sales, trace_forecast])

# Update layout
fig.update_layout(title='Actual Sales vs Forecast', xaxis_title='Index', yaxis_title='Sales')

# Show plot
fig.show()


# # now predict

# In[70]:


from pandas.tseries.offsets import DateOffset
import pandas as pd

# Convert index to datetime if it's not already in datetime format
df.index = pd.to_datetime(df.index)

# Generate future dates for prediction purpose
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, 48)]


# # DateOffset(months=x) for x in range(1, 48)] this 48 is no of months 
# 
# # you can give 12 ,24,36,48 anything

# In[71]:


#Convert that list into DATAFRAME:

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[72]:


future_datest_df.tail()


# In[73]:


#CONCATE THE ORIGINAL AND THE NEWLY CREATED DATASET FOR VISUALIZATION PURPOSE:
future_df=pd.concat([df,future_datest_df])


# In[83]:


#PREDICT
future_df['forecast'] = results.predict(start = 104, end = 151, dynamic= True)  
future_df[['sales', 'forecast']].plot(figsize=(12, 8))


# # future_df['forecast'] = results.predict(start = 104, end = 151, dynamic= True) 
# 
# # this 104 is the last row of the document of csv file 
# 
# # 151 is the total no.of rows i've after doing concat 

# In[84]:


import plotly.graph_objects as go

# Create traces
trace_actual_sales = go.Scatter(x=future_df.index, y=future_df['sales'], mode='lines', name='Actual Sales')
trace_forecast = go.Scatter(x=future_df.index, y=future_df['forecast'], mode='lines', name='Forecast')

# Create plotly figure
fig = go.Figure([trace_actual_sales, trace_forecast])

# Update layout
fig.update_layout(title='Actual Sales vs Forecast', xaxis_title='Index', yaxis_title='Sales')

# Show plot
fig.show()


# In[80]:


df.shape


# In[82]:


future_df.shape


# In[ ]:




