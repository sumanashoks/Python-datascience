#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

import warnings; 
warnings.simplefilter('ignore')


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[3]:


data


# In[4]:


sns.distplot(data['tip'])


# In[5]:


sns.distplot(data['total_bill'])


# In[6]:


kmeans=KMeans(n_clusters=4)
kmeans.fit(data.drop('gender','smoker','day','time','size',axis=1))


# # when we select more columns to drop we need to add inside[ ] square bracket.

# In[7]:


kmeans=KMeans(n_clusters=4)
kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'], axis=1))


# In[8]:


kmeans.cluster_centers_


# In[9]:


data['cluster_4']=kmeans.labels_
set(data['cluster_4'])


# In[10]:


data


# In[11]:


sns.scatterplot(x='total_bill',y='tip',hue='cluster_4',data=data)


# In[12]:


sns.lineplot(x='total_bill',y='tip',hue='cluster_4',data=data)


# In[13]:


sse={}

#For Loop to capture the Inertia

for k in range(1, 20):
    kmeans = KMeans(n_clusters=k).fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'],axis=1))
    sse[k] = kmeans.inertia_

print(sse)
#Store the no. of groups and the error as separate lists
groups=list(sse.keys())
error=list(sse.values())

#Club the lists as a dataframe
error_data= pd.DataFrame(
    {'groups': groups,
     'error': error
    })
error_data.head()


# In[16]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# Initialize an empty dictionary to store inertias
sse = {}

# Initialize KMeans object
kmeans = KMeans()

# Data is your dataset here

for k in range(1, 20):
    inertia_values = []
    for _ in range(10):  # Run KMeans multiple times for each k
        kmeans.set_params(n_clusters=k)
        kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'],axis=1))
        inertia_values.append(kmeans.inertia_)
    sse[k] = np.mean(inertia_values)  # Store average inertia for the given k

# Convert dictionary to lists
groups = list(sse.keys())
error = list(sse.values())

# Create a dataframe
error_data = pd.DataFrame({'groups': groups, 'error': error})

# Plotting
sns.pointplot(x="groups", y="error", data=error_data)


# In[14]:


sns.pointplot(x="groups", y="error", data=error_data)


# In[59]:


kmeans=KMeans(n_clusters=5)
kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size','cluster_4'], axis=1))


# In[60]:


kmeans.cluster_centers_


# In[61]:


data['cluster_5']=kmeans.labels_
set(data['cluster_5'])


# In[62]:


data


# In[76]:


sns.scatterplot(x='total_bill',y='tip',hue='cluster_5',data=data)


# In[64]:


sns.lineplot(x='total_bill',y='tip',hue='cluster_5',data=data)


# In[65]:


sns.scatterplot(x='total_bill',y='tip',hue='time',data=data)


# In[66]:


sns.boxplot(x='total_bill',y='tip',hue='time',data=data)


# In[67]:


sns.barplot(x='gender',y='tip',hue='time',data=data)


# In[68]:


sns.boxplot(x='gender',y='tip',hue='time',data=data)


# #  WHEN YOU ARE PLOTING GRAPHS REMEMBER THIS
# 
# #  ESPECIALLY WHEN YOU USE 'HUE'
#                |
#                |
#                |
#                |
#                |
#             \    /
#              \  /
#               \/
# 
#                

# # when inside a column if there any two names continously throughout the     column it will give accurate PLOTS
# 
#    # for example ---->1
#         # column  TIME
#         ----------------------
#                   DINNER
#                   LUNCH
#                   DINNER
#                   LUNCH         -----> # THIS COLUMN WILL GIVE ACCURATE BECAUSE IT HAD ONLY TWO VARIABLES
#                   DINNER
#                   LUNCH
#                 
#   # for example ----> 2
#       ## column.   CITY
#       --------------------------
#                    BLR
#                    MAA
#                    MUMBAI
#                    HYD    -------> # THIS COLUMN WILL GIVE MORE NUMBER OF BARS IT WIL BE DIFFICULT TO IDENTIFY
#                    KOLKATA
#                    DELHI
#                    PUNE
#                    AMEHADABAD
#                    KOCHI
#                    
#           
#                   
#                   
# 
# # when it more names then it is difficult to read.
# 

# # _______________________________________________________________________
# # _______________________________________________________________________

# In[74]:


plt.scatter(data['total_bill'],data['tip'])


# In[78]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')

kmeans = KMeans(n_clusters=4)
kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'], axis=1))

centroid_x = kmeans.cluster_centers_[:, 0]  # X coordinates of centroids
centroid_y = kmeans.cluster_centers_[:, 1]  # Y coordinates of centroids

data['cluster_4'] = kmeans.labels_

sns.scatterplot(x='total_bill', y='tip', hue='cluster_4', data=data)

# Plot centroids with '*' marker
plt.scatter(centroid_x, centroid_y, marker='*', color='RED', s=200)  # 's' sets the marker size

plt.show()


# In[81]:


kmeans.cluster_centers_


# In[ ]:





# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

data = pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[2]:


data


# In[10]:


scores_1 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(data.drop(['gender','smoker','day','time'],axis=1))
  scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()


# In[ ]:





# In[13]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'], axis=1))

centroid_x = kmeans.cluster_centers_[:, 0]  # X coordinates of centroids
centroid_y = kmeans.cluster_centers_[:, 1]  # Y coordinates of centroids

data['cluster_4'] = kmeans.labels_

sns.scatterplot(x='total_bill', y='tip', hue='cluster_4', data=data)

# Plot centroids with '*' marker
plt.scatter(centroid_x, centroid_y, marker='*', color='RED', s=200)  # 's' sets the marker size

plt.show()


# In[19]:


sse={}

#For Loop to capture the Inertia

for k in range(1, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'],axis=1))
    sse[k] = kmeans.inertia_

print(sse)
#Store the no. of groups and the error as separate lists
groups=list(sse.keys())
error=list(sse.values())

#Club the lists as a dataframe
error_data= pd.DataFrame(
    {'groups': groups,
     'error': error
    })
error_data.head()







sns.pointplot(x="groups", y="error", data=error_data)



# # always error should represnt in Y
# 
# # because always error rate will be in y axis in KNN
# 
# # then only you will get correct K bend

# In[20]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(data.drop(['gender', 'smoker', 'day', 'time', 'size'], axis=1))

centroid_x = kmeans.cluster_centers_[:, 0]  # X coordinates of centroids
centroid_y = kmeans.cluster_centers_[:, 1]  # Y coordinates of centroids

data['cluster_4'] = kmeans.labels_

sns.scatterplot(x='total_bill', y='tip', hue='cluster_4', data=data)

# Plot centroids with '*' marker
plt.scatter(centroid_x, centroid_y, marker='*', color='RED', s=200)  # 's' sets the marker size

plt.show()


# In[21]:


data.describe()


# In[26]:


data[data['tip']==10]


# In[28]:


data[data['tip']>5]


# In[31]:


data[data['total_bill']>45]


# In[33]:


print('missing values')
missing_values=data.isnull().sum()
print(missing_values)


# In[ ]:




