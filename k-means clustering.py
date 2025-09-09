#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[50]:


data=pd.read_csv('/Users/apple/Desktop/python/KMeans/KMeans.csv')


# In[51]:


data


# In[53]:


kmeans=KMeans(n_clusters=2)
kmeans.fit(data.drop('Driver_ID',axis=1))


# In[54]:


kmeans.cluster_centers_


# In[55]:


data['cluster_label_2']=kmeans.labels_
set(data['cluster_label_2'])


# In[56]:


data


# In[59]:


sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_2')


# # NOW 4 CLUSTERS WILL DO
# 

# In[66]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(data.drop(['Driver_ID','cluster_label_2'],axis=1))


# In[67]:


kmeans.cluster_centers_


# In[68]:


data['cluster_label_4']=kmeans.labels_
set(data['cluster_label_4'])


# In[69]:


sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_4')


# In[72]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(data.drop(['Driver_ID','cluster_label_2','cluster_label_4'],axis=1))

kmeans.cluster_centers_

data['cluster_label_5']=kmeans.labels_
set(data['cluster_label_5'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_5')


# In[73]:


kmeans.cluster_centers_


# In[74]:


kmeans = KMeans(n_clusters=6)
kmeans.fit(data.drop(['Driver_ID','cluster_label_2','cluster_label_4','cluster_label_5'],axis=1))

kmeans.cluster_centers_

data['cluster_label_6']=kmeans.labels_
set(data['cluster_label_6'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_6')


# In[75]:


kmeans = KMeans(n_clusters=8)
kmeans.fit(data.drop(['Driver_ID','cluster_label_2','cluster_label_4','cluster_label_5','cluster_label_6'],axis=1))

kmeans.cluster_centers_

data['cluster_label_8']=kmeans.labels_
set(data['cluster_label_8'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_8')


# In[77]:


kmeans = KMeans(n_clusters=10)
kmeans.fit(data.drop(['Driver_ID','cluster_label_2','cluster_label_4','cluster_label_5','cluster_label_6','cluster_label_8'],axis=1))

kmeans.cluster_centers_

data['cluster_label_10']=kmeans.labels_
set(data['cluster_label_10'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_10')


# # now i'm trying directly more clusters without droping unwanted columns

# In[78]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[80]:


suman=pd.read_csv('/Users/apple/Desktop/python/KMeans/KMeans.csv')


# In[81]:


suman


# In[82]:


kmeans = KMeans(n_clusters=8)
kmeans.fit(data.drop(['Driver_ID'],axis=1))

kmeans.cluster_centers_

data['cluster_label_8']=kmeans.labels_
set(data['cluster_label_8'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_8')


# In[83]:


kmeans = KMeans(n_clusters=10)
kmeans.fit(data.drop(['Driver_ID'],axis=1))

kmeans.cluster_centers_

data['cluster_label_10']=kmeans.labels_
set(data['cluster_label_10'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_10')


# In[85]:


kmeans = KMeans(n_clusters=10)
kmeans.fit(data.drop(['Driver_ID'],axis=1))


kmeans.cluster_centers_

data['cluster_label_10']=kmeans.labels_
set(data['cluster_label_10'])

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_10')


# # 1. the results are same when you drop the cloumns are not but resuls are              same
# 
# # 2. first i did 2 and 4 and 5 and 6 so i was dropping unwanted  columns  
# 
# #     for example
# 
#        ## when i wanted 4 cluster i dropped cluster 2
#        ## when i wanted 5 cluster i dropped cluster 2 and 4
#        ## when i wanted 6 cluster i dropped cluster 2,4 and 5 
#    
# # 3. YOU DON'T NEED TO DROP COLUMNS U CAN DROP ONLY ONE                                        COLUMN      THAT'S.  IT
# 
# 
# # 4.  WHY BECUASE -> 
#    
#    ## YOU ARE GIVING HUE AS CLUSTER NUMBER SO U DON'T NEED TO WORRY
#    
#    
#    
# 
# # 5.  I tried directly with higer number also the resilts are same

# # 6.  it's your wish u can drop or not

# # _________________________________________________________________________

# # ________________________________________________________________________
# 

# # now we'll do ELBOW method to find how many cluster can we do

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data=pd.read_csv('/Users/apple/Desktop/python/KMeans/KMeans.csv')


# In[4]:


data


# In[8]:


sse={}

#For Loop to capture the Inertia

for k in range(1, 20):
    kmeans = KMeans(n_clusters=k).fit(data.drop(['Driver_ID'],axis=1))
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


# In[9]:


kmeans=KMeans(n_clusters=5)
kmeans.fit(data.drop('Driver_ID',axis=1))


# In[10]:


kmeans.cluster_centers_


# In[11]:


data['cluster_label_5']=kmeans.labels_
set(data['cluster_label_5'])


# In[12]:


data


# In[14]:


sns.scatterplot(x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_5',data=data)


# In[15]:


sns.pointplot(x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_5',data=data)


# In[17]:


sns.jointplot(x='Distance_Feature',y='Speeding_Feature',hue='cluster_label_5',data=data)


# In[20]:


data.drop('Driver_ID',axis=1)
sns.pairplot(data=data,hue='cluster_label_5')


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/KMeans/KMeans.csv')


# In[3]:


data


# In[4]:


data.columns


# data.dtypes

# In[5]:


scores_1 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(data.drop(['Driver_ID'],axis=1))
  scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()


# In[6]:


kmeans=KMeans(n_clusters=5)
kmeans.fit(data.drop(['Driver_ID'],axis=1))

centroid_x=kmeans.cluster_centers_[:,0]
centroid_y=kmeans.cluster_centers_[:,1]

data['cluster_5']=kmeans.labels_

sns.scatterplot(data=data,x='Distance_Feature',y='Speeding_Feature',hue='cluster_5')

plt.scatter(centroid_x,centroid_y,marker='*',color='red',s=200)

plt.show()



# In[7]:


kmeans.cluster_centers_


# In[8]:


kmeans.cluster_centers_.shape


# In[11]:


plt.figure(figsize=(10,50))
for i in range(len(data.columns)):
  plt.subplot(17, 1, i+1)
  sns.distplot(data[data.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
  plt.title(data.columns[i])

plt.tight_layout()


# # PYSPARK

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
data = spark.read.csv("/Users/apple/Desktop/python/KMeans/KMeans.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=["Distance_Feature", "Speeding_Feature"], outputCol="features")
data = assembler.transform(data)

# Train KMeans model
kmeans = KMeans(k=4, seed=123)
model = kmeans.fit(data)

# Predict clusters
predictions = model.transform(data)

# Plot using seaborn
sns.scatterplot(data=predictions.toPandas(), x='Distance_Feature', y='Speeding_Feature', hue='prediction')
plt.show()

# Stop SparkSession
spark.stop()


# In[2]:


import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Create a SparkSession in local mode
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .master("local[*]") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
data = spark.read.csv("/Users/apple/Desktop/python/KMeans/KMeans.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=["Distance_Feature", "Speeding_Feature"], outputCol="features")
data = assembler.transform(data)

# Train KMeans model
kmeans = KMeans(k=4, seed=123)
model = kmeans.fit(data)

# Predict clusters
predictions = model.transform(data)

# Plot using seaborn
sns.scatterplot(data=predictions.toPandas(), x='Distance_Feature', y='Speeding_Feature', hue='prediction')
plt.show()

# Stop SparkSession
spark.stop()


# In[3]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Find and set the PySpark path
pyspark_submit_args = "--driver-class-path /path/to/spark/jars pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

# Create a SparkSession in local mode
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .master("local[*]") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
data = spark.read.csv("/Users/apple/Desktop/python/KMeans/KMeans.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=["Distance_Feature", "Speeding_Feature"], outputCol="features")
data = assembler.transform(data)

# Train KMeans model
kmeans = KMeans(k=4, seed=123)
model = kmeans.fit(data)

# Predict clusters
predictions = model.transform(data)

# Plot using seaborn
sns.scatterplot(data=predictions.toPandas(), x='Distance_Feature', y='Speeding_Feature', hue='prediction')
plt.show()

# Stop SparkSession
spark.stop()


# In[4]:


import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Find and set the PySpark path
pyspark_submit_args = "--driver-class-path /path/to/spark/jars pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

# Create a SparkSession in local mode
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .master("local[*]") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
data = spark.read.csv("/Users/apple/Desktop/python/KMeans/KMeans.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=["Distance_Feature", "Speeding_Feature"], outputCol="features")
data = assembler.transform(data)

# Train KMeans model
kmeans = KMeans(k=4, seed=123)
model = kmeans.fit(data)

# Predict clusters
predictions = model.transform(data)

# Plot using seaborn
sns.scatterplot(data=predictions.toPandas(), x='Distance_Feature', y='Speeding_Feature', hue='prediction')
plt.show()

# Stop SparkSession
spark.stop()



# In[5]:


from pyspark.sql import sparksession 


# In[6]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Create a SparkSession in local mode
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .master("local[*]") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
data = spark.read.csv("/Users/apple/Desktop/python/KMeans/KMeans.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=["Distance_Feature", "Speeding_Feature"], outputCol="features")
data = assembler.transform(data)

# Train KMeans model
kmeans = KMeans(k=4, seed=123)
model = kmeans.fit(data)

# Predict clusters
predictions = model.transform(data)

# Plot using seaborn
sns.scatterplot(data=predictions.toPandas(), x='Distance_Feature', y='Speeding_Feature', hue='prediction')
plt.show()

# Stop SparkSession
spark.stop()


# In[7]:


pip install pyspark


# In[8]:


import pyspark


# In[10]:


from pyspark.sql import SparkSession


# In[12]:


spark = SparkSession.builder.appName("KMeans Example").getOrCreate()


# In[ ]:




