#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[18]:


import pandas as pd

df = pd.read_csv('/Users/apple/Downloads/pandascleaning.csv')


# In[19]:


df


# In[20]:


df.dropna()


# In[21]:


df


# In[22]:


df.fillna(130)


# In[23]:


df


# In[24]:


df['calories'].fillna(130)


# In[25]:


df


# In[28]:


x = df['calories'].mean()

df['calories'].fillna(x)


# In[29]:


df


# In[33]:


x = df['calories'].median()

df['calories'].fillna(x)


# In[34]:


x = df['calories'].mode()[0]

df['calories'].fillna(x)


# In[35]:


df


# In[36]:


df.loc[4,'duration']=60


# In[37]:


df


# In[38]:


df


# In[41]:


for x in df.index:
  if df.loc[x, 'duration'] > 50:
    df.loc[x, 'duration'] = 40


# In[42]:


df


# In[45]:


print(df.duplicated())


# In[46]:


df.drop_duplicates()


# In[ ]:




