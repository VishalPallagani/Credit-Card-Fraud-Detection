
# coding: utf-8

# In[6]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))


# In[7]:


#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#loading dataset
data = pd.read_csv('creditcard.csv')


# In[9]:


#exploring the dataset
print(data.columns)


# In[10]:


print(data.shape)


# In[11]:


print(data.describe)


# In[12]:


print(data.describe())


# In[13]:


data = data.sample(frac=0.1, random_state = 1)
print(data.shape)


# In[14]:


#Plotting histogram of each param
data.hist(figsize = (20,20))
plt.show()


# In[15]:


#Determine number of fraud cases in dataset
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Cases: {}'.format(len(Valid)))


# In[16]:


#Correlation matrix
corrmat = data.corr();
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[17]:


#All columns from data frame
columns = data.columns.tolist()

#Filtering columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

#Store the variable we will be prdeciting on
target = "Class"

X=data[columns]
Y=data[target]

#Printing the shapes of X and Y
print(X.shape)
print(Y.shape)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#defining a random state
state = 1

#defining outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X), contamination = outlier_fraction, random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}


# In[21]:


#fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # Reshaping the prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #Running classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))

