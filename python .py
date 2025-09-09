#!/usr/bin/env python
# coding: utf-8

# In[3]:


a=1
b=2
c=a+b
print("c:",+ "suman")


# In[8]:


rcb= virat kholi
print("rcb"[1:3])


# In[11]:


rcb=="win"
if(rcb=="win"):
    print(ESCN)
esle:
    print(NSCN)


# In[ ]:





# In[12]:


LOSE


# In[13]:


RCB= "VIRAT KHOLI"
print(RCB[0])


# In[2]:


rcb="win"
if(rcb=="win"):
    print("rcb won")
esle:
    print("nope")


# In[ ]:


rcb="win"
if(rcb=="win"):
    print("rcb won")
else:
    print("nope")


# In[ ]:


rcb=("win")
if(rcb=="win"):
    print("rcb won")
else:
    print("nope")


# In[ ]:





# In[3]:


import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45],
  "steps"   : [12,20,30]
}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df.loc[[0,1,2]])


# In[26]:


import pandas as pd

score = {
    "india": [100, 100, 300, 400],
    "england": [100, 20, 30, 90],
    "south_africa": [10, 100, 99, 50],
}

df = pd.DataFrame(score, index=[1,2,3,4])

print(df)


# In[27]:


import pandas as pd

score = {
    "india": [100, 100, 300, 400],
    "england": [100, 20, 30, 90],
    "south_africa": [10, 100, 99, 50],
}

df = pd.Series(score)

print(df)


# In[28]:


df.loc([0])


# In[2]:


import pandas as pd

score ={
    'india':[12,13,14,15],
    'pak':[11,12,11,15]
}
df=pd.DataFrame(score, index=[1,2,3,4])
print(df)


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


from numpy.random import randn


# In[11]:


df = pd.DataFrame(randn(5,5),index=['A','B','C','D','E'],columns=['v','W','X','Y','Z',])


# In[12]:


df


# In[29]:


df[['W','Z','Y']]


# In[31]:


df.W


# In[ ]:




