#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from numpy.random import randn


# In[4]:


df=pd.DataFrame(randn(5,5),index=['a','b','c','d','e'],columns=['w','x','y','z','q'])


# In[5]:


df


# In[6]:


df['w']


# In[7]:


df[['w','x']]


# In[8]:


df


# In[9]:


df['new']=df['w']+df['x']


# In[10]:


df


# In[11]:


df.drop('new',axis=1,inplace=True)


# In[13]:


df


# In[14]:


df.loc['a']


# In[15]:


df.loc[['a','b']]


# In[16]:


df


# In[17]:


df.loc[['a','b'],['w','x']]


# In[18]:


df


# In[19]:


df>0


# In[22]:


df[df>0]


# In[33]:


df


# In[34]:


df[df>0]


# In[35]:


df = df.fillna(0)


# In[36]:


df[df>0]


# In[39]:


import pandas as pd
import numpy as np


# In[43]:


df1=pd.DataFrame({
    'A':['A0', 'A1', 'A2', 'A3'],
    'B':['B0', 'B1', 'B2', 'B3'],
    'C':['C0', 'C1', 'C2', 'C3'],
    'D':['D0', 'D1', 'D2', 'D3']},
    index=[0, 1, 2, 3]
)


# In[45]:


df2=pd.DataFrame({
    'A':['A4', 'A5', 'A6', 'A7'],
    'B':['B4', 'B5', 'B6', 'B7'],
    'C':['C4', 'C5', 'C6', 'C7'],
    'D':['D4', 'D5', 'D6', 'D7']
},index=[4,5,6,7])


# In[46]:


df3=pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])


# In[47]:


df1


# In[48]:


df2


# In[49]:


df3


# In[50]:


pd.concat([df1,df2,df3])


# In[51]:


left=pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']
})


# In[53]:


right=pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})


# In[54]:


left


# In[55]:


right


# In[67]:


pd.merge(left,right,on='key')


# In[68]:


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})


# In[69]:


pd.merge(left,right,on=['key1','key2'])


# In[71]:


pd.merge(left,right,how='outer',on=['key1','key2'])


# In[72]:


pd.merge(left,right,how='right',on=['key1','key2'])


# In[73]:


pd.merge(left,right,how='left',on=['key1','key2'])


# In[75]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


# In[76]:


left.join(right)


# In[77]:


left.join(right,how='outer')


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[5]:


data = pd.read_csv(r'/Users/apple/Desktop/python/KNN/advertising.csv')



# In[6]:


data


# In[ ]:





# In[ ]:




