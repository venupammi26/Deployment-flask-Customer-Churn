#!/usr/bin/env python
# coding: utf-8

# In[162]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns #for background style for plots
from matplotlib.pyplot import figure #plot size
import matplotlib.style #Setting the theme of your plots

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows',100)


# # Data Collection

# In[208]:


data=pd.read_csv("TRAIN.csv")
print(data.shape)
data.head()


# # EDA & Preprocessing

# In[209]:


data.columns=data.columns.str.replace(' ', '')


# In[210]:


#Missing Values identifying

data.isnull().sum()


# Replace null values in these columns with 'Other'

# In[211]:


data['NetworktypesubscriptioninMonth2'].fillna('Other', inplace = True)
data['NetworktypesubscriptioninMonth1'].fillna('Other', inplace = True)
data.isnull().sum()


# Remove remaining null values
# 

# In[212]:


data=data.dropna()
data.isnull().sum()


# In[213]:


#Categorical Column

cat_cols=data.select_dtypes(exclude=['int64', 'float']).columns[1:]
cat_cols


# In[214]:


#Create categorical target variable for chi test
data['Target']=data['ChurnStatus'].apply(lambda x: 'No' if x==0.0 else 'Yes')
data.head()


# In[215]:


cat_cols


# In[216]:


import scipy.stats
rec=[]
for col2 in cat_cols:
        df=pd.crosstab(data['Target'],data[col2])
        chi= scipy.stats.chi2_contingency(df)
        if chi[1]<=0.05:
            print(col2 ,"  = " , chi[1])
            


# Target variable is dependent on these three categorical variables

# In[217]:


data1=data.drop(['Target','NetworktypesubscriptioninMonth1','CustomerID'] ,axis=1)


# In[218]:


#One Hot Encoding

from sklearn.preprocessing import OneHotEncoder

data2 = pd.get_dummies(data1)


# In[219]:


cr = data2.corr()
cr.ChurnStatus


# In[220]:


#Data Partition

X = data2.drop('ChurnStatus', axis =1)
y = data2['ChurnStatus']


# In[221]:


#Data SPlitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# # Model Training
# 

# # Gradient Boost & Classifier

# In[222]:


from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
GBC=GradientBoostingClassifier(n_estimators=150)
GBC.fit(X_train,y_train)
y_pred = GBC.predict(X_test)


print('Confusion matrix:  \n', confusion_matrix(y_test,y_pred))
print('accuracy_score   :   ', accuracy_score(y_test,y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# Lets check     important features

# In[223]:


print(GBC.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(GBC.feature_importances_, index=X.columns)
feat_importances.nlargest(8).plot(kind='barh')
plt.show()


# Lets test with these features

# In[224]:


X2= X[feat_importances.nlargest(8).index]

X2_train,X2_test,y_train,y_test = train_test_split(X2,y,test_size=0.3,random_state=1)


# In[225]:


feat_importances.nlargest(8).index


# In[226]:


GBC.fit(X2_train,y_train)
y_pred = GBC.predict(X2_test)


print('Confusion matrix:  \n', confusion_matrix(y_test,y_pred))
print('accuracy_score   :   ', accuracy_score(y_test,y_pred))

fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# These are important features and This is the final model

# In[227]:


#So i have to save model
import pickle
pickle.dump(GBC, open('model.pkl','wb'))



# In[ ]:




