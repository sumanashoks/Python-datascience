#!/usr/bin/env python
# coding: utf-8

# In[1]:


from alpha_vantage.timeseries import TimeSeries


# In[2]:


API_KEY='VA7Q2MQAMSG4ZG4G'


# In[3]:


ts=TimeSeries(key=API_KEY,output_format='pandas')


# In[4]:


data=ts.get_weekly('MSFT')

data[0]


# In[5]:


data=ts.get_intraday('MSFT',interval='15min')

data[0]


# In[6]:


from alpha_vantage.fundamentaldata import FundamentalData

# Assuming 'key' holds your API key
key = 'VA7Q2MQAMSG4ZG4G'

# Creating an instance of FundamentalData
fd = FundamentalData(key, output_format='pandas')

# Now, you can use 'fd' to access the methods provided by FundamentalData
# For example:
# fd.get_company_overview(symbol='AAPL')
# This would retrieve fundamental data for the company with the symbol 'AAPL'


# In[20]:


data=fd.get_balance_sheet_annual(symbol='AAPL')

data[0].T

#T represents the transprose


# In[7]:


from alpha_vantage.timeseries import TimeSeries

key = 'VA7Q2MQAMSG4ZG4G'
outputsize = 'compact'
symbol = input('Ticker: ')
typ = input('Data type - "daily", "weekly", "monthly", "interval": ')

ts = TimeSeries(key, output_format='pandas')  # Corrected line

if typ == 'daily':
    state = ts.get_daily_adjusted(symbol, outputsize=outputsize)[0]
elif typ == 'weekly':
    state = ts.get_weekly_adjusted(symbol)[0]
elif typ == 'monthly':
    state = ts.get_monthly_adjusted(symbol)[0]
elif typ == 'interval':
    interval = input('Interval - 1min, 5min, 15min, 30min, 60min: ')
    state = ts.get_intraday(symbol, interval=interval, outputsize=outputsize)[0]
else:
    print('Wrong entry')

state  # This will display the state variable, adjust this based on your requirement


# In[8]:


from alpha_vantage.timeseries import TimeSeries

key = 'VA7Q2MQAMSG4ZG4G'
outputsize = 'compact'
symbol = input('Ticker: ')
typ = input('Data type - "daily", "weekly", "monthly", "interval": ')

ts = TimeSeries(key, output_format='pandas')  # Corrected line

if typ == 'daily':
    state = ts.get_daily_adjusted(symbol, outputsize=outputsize)[0]
elif typ == 'weekly':
    state = ts.get_weekly_adjusted(symbol)[0]
elif typ == 'monthly':
    state = ts.get_monthly_adjusted(symbol)[0]
elif typ == 'interval':
    interval = input('Interval - 1min, 5min, 15min, 30min, 60min: ')
    state = ts.get_intraday(symbol, interval=interval, outputsize=outputsize)[0]
else:
    print('Wrong entry')

state  # This will display the state variable, adjust this based on your requirement


# In[ ]:


from alpha_vantage.fundamentaldata import FundamentalData
key = 'VA7Q2MQAMSG4ZG4G'
symbol = input('Ticker: ')
period = input('Period - annual, quarterly: ')
statement = input('Statement - balance sheet, income statement, cash flow: ')

fd = FundamentalData(key, output_format='pandas')

if period == 'annual':
    if statement == 'balance sheet':
        state = fd.get_balance_sheet_annual(symbol)[0].T[2:]
        state.columns = list(fd.get_balance_sheet_annual(symbol)[0].T.iloc[0])
    elif statement == 'income statement':
        state = fd.get_income_statement_annual(symbol)[0].T[2:]
        state.columns = list(fd.get_income_statement_annual(symbol)[0].T.iloc[0])
    elif statement == 'cash flow':
        state = fd.get_cash_flow_annual(symbol)[0].T[2:]
        state.columns = list(fd.get_cash_flow_annual(symbol)[0].T.iloc[0])
    else:
        print('Wrong Entry')
elif period == 'quarterly':
    if statement == 'balance sheet':
        state = fd.get_balance_sheet_quarterly(symbol)[0].T[2:]
        state.columns = list(fd.get_balance_sheet_quarterly(symbol)[0].T.iloc[0])
    elif statement == 'income statement':
        state = fd.get_income_statement_quarterly(symbol)[0].T[2:]
        state.columns = list(fd.get_income_statement_quarterly(symbol)[0].T.iloc[0])
    elif statement == 'cash flow':
        state = fd.get_cash_flow_quarterly(symbol)[0].T[2:]
        state.columns = list(fd.get_cash_flow_quarterly(symbol)[0].T.iloc[0])
    else:
        print('Wrong Entry')
else:
    print('Wrong Entry')

state



# In[ ]:




