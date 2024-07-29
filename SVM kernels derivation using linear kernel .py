#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


x = np.linspace(-5.0,5.0,100)
y = np.sqrt(10**2 - x**2)

x


# In[23]:


y


# In[24]:


y = np.hstack([y,-y])
x= np.hstack([x, -x])


# In[25]:


x1 = np.linspace(-5.0,5.0,100)
y1 = np.sqrt(5**2 - x1**2)
y1 = np.hstack([y1,-y1])
x1 = np.hstack([x1, -x1])


# In[26]:


plt.scatter(y,x)
plt.scatter(y1,x1)


# In[27]:


df1 = pd.DataFrame(np.vstack([y,x]).T, columns=['X1','X2'])
df1['Y'] =0
df2= pd.DataFrame(np.vstack([y1,x1]).T, columns=['X1','X2'])
df2['Y'] =1
frames= [df1,df2]
df = pd.concat(frames)
df


# In[28]:


#dependent and independent features
X = df.iloc[:, :2]
y = df['Y']


# In[11]:


X


# In[29]:


y


# In[30]:


# X1_square , X2_squre , X1X2
df['X1_square'] = df['X1']**2
df['X2_square'] = df['X2']**2
df['X1X2']= df['X1']*df['X2']
df


# In[32]:


X = df[['X1','X2','X1_square', 'X2_square','X1X2']]
y = df['Y']


# In[33]:


#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[34]:


X_train


# In[35]:


y_train


# In[37]:


import plotly.express as px
px.scatter_3d(df, x='X1_square', y='X2_square', z='X1X2')


# In[41]:


from sklearn.svm import SVC
model_name = SVC(kernel='linear')
model_name.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = model_name.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:




