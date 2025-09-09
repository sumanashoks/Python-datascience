#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from numpy.random import randn


# In[5]:


df = pd.DataFrame(randn(5,5),index=['A','B','C','D','E'],columns=['suman','charan','chandan','ajay','chethan'])


# In[6]:


df


# In[9]:


df[['suman','charan']]


# In[10]:


df


# In[15]:


df.suman


# In[16]:


df['jatin']=df['chandan']+df['chethan']


# In[17]:


df


# In[19]:


df.drop('jatin',axis=1)


# In[20]:


df


# In[21]:


df.drop('jatin',axis=1,inplace=True)


# In[22]:


df


# In[23]:


df.drop('D',axis=0) # axis=1 represents the column and axis=0 represents rows


# In[24]:


df


# In[27]:


df


# In[28]:


df.drop('D',axis=0,inplace=True)


# In[29]:


df


# In[30]:


df.suman


# In[31]:


df[['suman','charan']]


# In[34]:


df.loc[['A','B']]       # LOC IS USED FOR ROWS TO FIND OUT 


# In[39]:


df.loc['A','suman']   # i selected both row and column "a" is row and "suman" is column


# In[37]:


df.loc[['A','B'],['suman','charan']]


# In[40]:


df


# In[41]:


df>0


# In[45]:


df[df>0]


# In[49]:


df[df>0]


# In[50]:


df[df['suman']>0]['charan']


# In[ ]:




