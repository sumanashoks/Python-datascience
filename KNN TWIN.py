#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


ad_data=pd.read_csv('/Users/apple/Desktop/python/KNN/advertising.csv')


# In[3]:


ad_data.head(n=10)


# In[4]:


ad_data.describe()


# In[15]:


ad_data.head()


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assume ad_data is your DataFrame with the data

# Split the data into features and target
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Elbow method to determine the optimal K value
error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1, 20), error_rate)
plt.title('Cross-Validation Scores for Different K Values')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate')
plt.show()


# In[6]:


X.shape,y.shape,X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[7]:


import pandas as pd

# Assuming X, y, X_train, X_test, y_train, and y_test are your arrays
data = {
    'Array': ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test'],
    'Shape': [X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape]
}

df = pd.DataFrame(data)
df


# In[8]:


import pandas as pd

# Assuming X, y, X_train, X_test, y_train, and y_test are your arrays
data = {
    'Array': ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test'],
    'Shape': [X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape]
}

df = pd.DataFrame(data)

# Add styling for borders
styled_df = df.style.set_table_styles([{'selector': 'table', 'props': [('border', '1px solid black')]}, 
                                       {'selector': 'th', 'props': [('border', '1px solid black')]}, 
                                       {'selector': 'td', 'props': [('border', '1px solid black')]}])

# Display the styled DataFrame
styled_df


# In[9]:


from sklearn.metrics import classification_report,confusion_matrix
#Build the Model with 2 as K
knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[24]:


from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the KNN classifier
knn = KNeighborsClassifier()

# Assume 'X_train' is your training data and 'y_train' are the corresponding labels
# Replace 'X_train' and 'y_train' with your actual training data and labels
knn.fit(X_train, y_train)

# Now you can use the 'knn' object to make predictions
predictions = knn.predict([[100, 36, 60000, 250, 0]])
print(predictions)


# In[16]:


ad_data.head(n=2)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))


# # Exporting as CSV

# In[11]:


# Convert scaled arrays back to a DataFrame
scaled_data = pd.DataFrame(X_train, columns=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male'])

# Export scaled data to CSV
scaled_data.to_csv('/Users/apple/Desktop/python/KNN/scaled_ad_data.csv', index=False)


# In[12]:


ad_data.dtypes


# In[13]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming you have already trained your decision tree model and named it 'dtree'

feature_names = list(X.columns.get_level_values(0))     # Extracting feature names from MultiIndex
plt.figure(figsize=(20, 10))                                 # Adjust the figure size as needed
plot_tree(dtree, filled=True, feature_names=feature_names)  # Plotting the decision tree
plt.title("Decision Tree Plot")
plt.show()


# In[14]:


knn.score(X_test,y_test)


# In[ ]:





# In[ ]:




