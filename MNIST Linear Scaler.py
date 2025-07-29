#!/usr/bin/env python
# coding: utf-8

# # DELETE ME 
# (Ch 5 Exercise 9) Train an SVM classifier on the MNIST dataset or an appropriate dataset of your choice. Since SVM classifiers are binary classifiers, you will need to use one-versus-the-rest to classify all 10 digits. You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?
# 
# ## Ask questions

# # MNIST
# 
# 
# ## WRITE SOME NONSENSE HERE!

# ## Importing the data
# Luckilly the MNIST dataset can be found in the SKLearn data set module, not to far out of the way.  

# In[1]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


# Lets look inside of one of the elements.  Can you tell what number it looks like? Me neither, lets utitlize some data visulization tools to see what number we are looking at.  

# In[2]:


mnist['data'][0]


# I'm calling it, that's a 5.  Using simple data vis techneques takes a lot of guesswork out of this.

# In[3]:


import matplotlib.pyplot as plt
fig = plt.figure
plt.imshow(mnist['data'][0].reshape(28,28),cmap =plt.cm.gray_r, interpolation = "nearest") 
plt.show()


# In[4]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


# ## Assign Train and Test Data
# There appear to be 7 thousand total rows in the dataset.  We want to keep 1000 rows of test data.  Hence, we will split the data file up by 1/7. 

# In[5]:


X = mnist["data"]
y = mnist["target"]
from sklearn.model_selection import train_test_split 
X_train, X_test = train_test_split(X, test_size=(1/7), random_state=42)
y_train, y_test = train_test_split(y, test_size=(1/7), random_state=42)


# ## Creating the model 
# This has got to  be a class record for how lon it takes to prepare the data for modeling.  Which gives us more time to budget building the best model possible.  We can creat our first iteration of a OvR model.  

# In[ ]:


from sklearn.svm import LinearSVC
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train, y_train)


# However, this iteration came quickly, we got not so good results.  Luckilly, we left pleanty of room to adjust the data and the model.

# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = lin_clf.predict(X_train)
accuracy_score(y_train, y_pred)


# ## Scaling the data
# The data may have unnecessary inconsistancies that need to be scaled.  This will likely eliminate some noise and allow the model to focus on patterns that are necessary. That is, we reduce the number of rules required for the model.

# In[ ]:


import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))


# This appeared to help our score, but we certianly need to find other ways to improve our methods in order to produce a model worth using.  

# In[ ]:


lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train_scaled, y_train)
y_pred = lin_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)


# ### Simple Parameter Adjustments
# There we go, now that we have adjusted the parameters, and seem to have promising results.  

# In[ ]:


from sklearn.svm import SVC
svm_clf = SVC(decision_function_shape="ovr", gamma="auto")
svm_clf.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = svm_clf.predict(X_train_scaled)
accuracy_score(y_train, y_pred)


# If we look into more detail at the confusion matrix, we see that we are:
# * best at classifying 1s
# * Worst at classifying 5s
# * The most common error is confusing a 7 for a 9

# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_pred)


# ## Hyper Parameter tuning.  
# We are getting close to our final answer, we hope to make a models that not only outpreforms humans at speed, but can compete with accuracy. 

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


# In[ ]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:





# In[ ]:


cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[ ]:


print(grid.best_params_)
print(grid.best_estimator_)


# In[ ]:




