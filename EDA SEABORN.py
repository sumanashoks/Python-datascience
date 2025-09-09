#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib

print(matplotlib.__version__)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'sans-serif','color':'blue','size':20}
font2 = {'family':'cursive','color':'darkred','size':15}

plt.title("Sports Watch Data",loc='center', fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.bar(x, y)
plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.grid()



plt.show()


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[3]:


data


# In[4]:


data.head(n=10)


# In[5]:


data.info()


# In[26]:


data['day'].value_counts(normalize=True) * 100
#In this by executing the above code we will get to know how many percentage for each day and how many days are there in this


# In[27]:


data['gender'].value_counts(normalize=True) * 100


# In[28]:


data['time'].value_counts(normalize=True) * 100


# In[29]:


data['size'].value_counts(normalize=True) * 100


# In[32]:


data['time'].unique()
#this is used for to know how many values arethere in a column uniquely


# In[33]:


print('missing values check')
missing_values=data.isnull().sum()
print(missing_values)


# In[34]:


sns.pairplot(data,hue='time')
plt.title('suman')
plt.show()


# In[37]:


data['total_spend']=data['total_bill']+data['tip']


# In[43]:


data


# In[42]:





# In[10]:


sns.barplot(x='gender', y='tip', hue='time', data=data)

plt.title('Tips by Gender and time of dining')
plt.show()


# In[11]:


sns.barplot(x='gender',y='tip',data=data,hue='time')
plt.title('gender tips by time')
plt.show()


# In[12]:


plt.figure(figsize=(5,4))
sns.barplot(x='gender',y='tip',data=data,)
plt.title('gender tips by time')             #this is without hue
plt.show()


# In[44]:


plt.figure(figsize=(10,5))
sns.histplot(data['total_spend'],kde=True,bins=20,color='red')
plt.show()


# # YOU CAN USE HEAT MAP FOR ALL NUMBER COLUMNS

# In[45]:


numeric_columns=data.select_dtypes(include='number')
correlation_matrix=numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()


# # YOU CAN US HEAT MAP ONLY FOR PARTICULAR COLUMNS

# In[47]:


data=data[['tip','total_spend']]
correlation_matrix=data.corr() 
sns.heatmap(correlation_matrix, annot=True)


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



data=pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[2]:


data['total_spend']=data['total_bill']+data['tip']


# In[3]:


data


# In[4]:


#violinplot gives the exact answer but barplot gives little inaccurate
sns.boxplot(x='gender',y='tip',data=data,hue='time') 
plt.show()


# In[ ]:





# In[5]:


sns.lineplot(x='total_bill', y='total_spend', data=data, hue='time')
plt.show()


# In[6]:


sns.pairplot(data,hue='total_spend')
plt.show()


# In[7]:


sns.distplot(data['total_bill'],kde=True,bins=20)
plt.grid()


# In[8]:


data


# # ______________________________________________________________________

# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


df=pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')


# In[11]:


df


# In[12]:


df['total_spend']=df['total_bill']+df['tip']


# In[13]:


df


# # Explorative Data Analysis

# In[14]:


df.head(n=5)


# In[15]:


df.tail(n=5)


# In[16]:


df.shape


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


df.groupby('day').describe()


# In[20]:


df['day'].unique()


# In[21]:


df['day'].value_counts(normalize=True)


# In[22]:


df.isnull().sum()


# In[23]:


df.dropna(subset=['tip'],inplace=True)


# In[24]:


df['tip'].fillna(df['tip'].mean())


# In[25]:


df.dtypes


# In[26]:


df.columns


# In[27]:


df.describe(include='object') # it shows the string column object represents the string column


# In[28]:


for col in df.describe(include='object').columns:
    print(col)
    print(df[col].unique())
    print('_'*50)


# In[29]:


df.describe()


# In[30]:


df[df['total_bill']==50.810000]


# In[31]:


df=df[df['total_bill']<50] # removing outliers because 50 was dominating in total_bill column


# In[32]:


df.describe()


# In[33]:


df=df[df['total_bill']<40]


# In[34]:


df.describe()


# In[35]:


sns.boxplot(data=df)


# ##  there is a outlier in the above graph

# In[36]:


# Removing values above 35 and below 2 in the 'your_column' column
df = df[(df['total_bill'] <= 35) & (df['total_bill'] >= 2)]


# In[37]:


df.describe()


# In[38]:


sns.boxplot(data=df)


# ## I removed the outlier just look the above graph

# In[39]:


df['smoker'].value_counts(normalize=True)*100


# In[40]:


df


# In[41]:


sns.barplot(x=df['gender'].value_counts().index, y=df['gender'].value_counts())
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Count of Gender')
plt.show()


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'gender' and 'time' are columns in your DataFrame
sns.countplot(x='gender', hue='time', data=df)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Count of Gender')
plt.show()


# In[43]:


sns.countplot(x='gender',hue='time',data=df)


# In[44]:


sns.barplot(x='gender',y='tip',hue='time',data=df)


# In[45]:


sns.countplot(x='smoker',data=df)


# In[46]:


sns.barplot(x='day',y='tip',data=df)


# In[47]:


sns.barplot(x='gender',y='tip',data=df,hue='time')


# In[48]:


sns.barplot(x='gender',y='total_spend',data=df)


# In[49]:


sns.violinplot(x='gender',y='total_spend',data=df)


# In[53]:


import dtale

dtale.show(df)


# In[51]:


import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('/Users/apple/Desktop/python/DS Visualization/tips.csv')

df['total_spend']=df['total_bill']+df['tip']

profile = ProfileReport(df, title="Profiling Report")
profile.to_file('output3.html')




# In[52]:


df


# In[55]:


df['day'].value_counts(normalize=True)*100


# In[58]:


SAT=df[df['day']=='Sat']
SAT['gender'].value_counts(normalize=True)*100

# I GAVE A VARIABLE NAME SAT to find HOW MANY MALE AND FEMALE GO ON SATURDAY
# AND I PULLED DAY COLUMN AND SPECIFIED SATURDAY
# NOW SAT STORES THE SATURDAY DETAILS
# NOW PULL SAT AND SPECIFY GENDER TO KNOW HOW MANY MALE & FEMALE  GO ON SATURDAY 


# In[59]:


SUN=df[df['day']=='Sun']
SUN['gender'].value_counts(normalize=True)*100


# In[60]:


smoking=df[df['smoker']=='Yes']
smoking['gender'].value_counts(normalize=True)*100


# In[61]:


smoking=df[df['smoker']=='No']
smoking['gender'].value_counts(normalize=True)*100


# In[62]:


df['smoker'].value_counts(normalize=True)*100


# In[ ]:




