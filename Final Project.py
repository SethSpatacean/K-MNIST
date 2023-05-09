#!/usr/bin/env python
# coding: utf-8

# # Final Project: Kuzushiji-MNIST
# ### Seth Spatacean ID: 15939018
# ---

# ## Task 1: Loading the Data

# In[18]:


import pandas as pd
import numpy as np
kmnist_train = pd.read_csv('kmnist_train.csv')
kmnist_test = pd.read_csv('kmnist_test.csv')
kmnist_train1 = kmnist_train.drop(columns = 'Unnamed: 0') # drop unnamed column from datasets
kmnist_test1 = kmnist_test.drop(columns = 'Unnamed: 0')


# In[19]:


X_train = kmnist_train1.drop('label',axis = 1)
X_train = X_train.to_numpy()
X_test = kmnist_test1.drop('label', axis = 1)
X_test = X_test.to_numpy()

y_train = kmnist_train1['label'].to_numpy()  
y_test = kmnist_test1['label'].to_numpy()


# In[20]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ---

# ## Task 2: Logistic Regression

# In[21]:


import numpy as np

class myLogisticRegression():
    """ Logistic Regression classifier -- this also works for the multiclass case.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    """
    def __init__(self, learning_rate=.001, opt_method = 'SGD', num_epochs = 50, size_batch = 20):
        
        # learning rate can also be in the fit method
        self.learning_rate = learning_rate
        self.opt_method = opt_method
        self.num_epochs = num_epochs
        self.size_batch = size_batch
        

    def fit(self, data, y, n_iterations = 1000):
        """ 
        fit function:
        - trains the dataset
        - uses stochastic gradient descent loss function to update coefficients (W)
        in order to obtain a higher accuracy of predictions
        """
        self.K = max(y)+1 # specify number of classes in y
        ones = np.ones((data.shape[0],1)) # column of ones 
        X = np.concatenate((ones, data), axis = 1) # the augmented matrix, \tilde{X} in our lecture
        eta = self.learning_rate
        
        W  = np.zeros((np.shape(X)[1],max(y)+1))
        
        if self.opt_method == 'GD':
            for k in range(n_iterations):
                dW = self.loss_gradient(W,X,y) # write another function to compute gradient
                W = W - eta * dW # the formula of GD
                # this step is optional -- just for inspection purposes
                if k % 500 == 0: # pprint loss every 50 steps
                    print("loss after", k+1, "iterations is: ", self.loss(W,X,y))
        
        if self.opt_method == 'SGD':
            N = X.shape[0]
            num_epochs = self.num_epochs
            size_batch = self.size_batch
            num_iter = 0
            for e in range(num_epochs):
                shuffle_index = np.random.permutation(N) # in each epoch, we first reshuffle the data to create "randomness"
                for m in range(0,N,size_batch):   # m is the starting index of mini-batch
                    i = shuffle_index[m:m+size_batch] # index of samples in the mini-batch
                    dW = self.loss_gradient(W,X[i,:],y[i]) # only use the data in mini-batch to compute gradient. Note the average is taken in the loss_gradient function
                    W = W - eta * dW # the formula of GD, but this time dbeta is different
                
                    if e % 1 == 0 and num_iter % 50 ==0: # print loss during the training process
                        print("loss after", e+1, "epochs and ", num_iter+1, "iterations is: ", self.loss(W,X,y))
        
                    num_iter = num_iter +1  # number of total iterations
            
        self.coeff = W
        
    def predict(self, data):
        '''
        predict function:
        - Takes argument X from either train or test data
        - Makes a prediction of y (label) accoring to the fit method which updates W
        '''
        ones = np.ones((data.shape[0],1)) # column of ones 
        X = np.concatenate((ones, data), axis = 1) # the augmented matrix, \tilde{X} in our lecture
        W = self.coeff # the estimated W
        y_pred = np.argmax(self.sigma(X,W), axis = 1) # the category with largest probability
        return y_pred
    
    def score(self, data, y_true):
        '''
        score function:
        - determies model accuracy
        - with inputs X_test, y_test; 
        the score function compares the predictions of the model against the real labels
        '''
        ones = np.ones((data.shape[0],1)) # column of ones 
        X = np.concatenate((ones, data), axis = 1) # the augmented matrix, \tilde{X} in our lecture
        y_pred = self.predict(data)
        acc = np.mean(y_pred == y_true) # number of correct predictions/N
        return acc
    
    def sigma(self,X,W): #return the softmax probability
        '''
        sigma function: (softmax)
        - normalizes the inputs into a probability distribution
        - output represents a categorical distribution
        '''
        s = np.exp(np.matmul(X,W))
        total = np.sum(s, axis=1).reshape(-1,1)
        return s/total
    
    def loss(self,W,X,y):
        '''
        loss function:
        - a method of evaluating the model in its ability to represent the dataset
        - determines what a poor prediction and a good prediction is
        - can represent the 'error' with respect to the training dataset 
        '''
        f_value = self.sigma(X,W)
        K = self.K 
        loss_vector = np.zeros(X.shape[0])
        for k in range(K):
            loss_vector += np.log(f_value+1e-10)[:,k] * (y == k) # avoid nan issues
        return -np.mean(loss_vector)
                          
    def loss_gradient(self,W,X,y):
        '''
        loss_gradient function:
        - computes the gradient of the loss function
        - represents the slope,or how much we move our weights by in order to minimize error
        - used in the fit() function for the gradient descent method
        '''
        f_value = self.sigma(X,W)
        K = self.K 
        dLdW = np.zeros((X.shape[1],K))
        for k in range(K):
            dLdWk =(f_value[:,k] - (y==k)).reshape(-1,1)*X # Numpy broadcasting
            dLdW[:,k] = np.mean(dLdWk, axis=0)   # RHS is 1D Numpy array -- so you can safely put it in the k-th column of 2D array dLdW
        return dLdW


# In[22]:


'''Evaluate Performance of myLogisticRegression'''

mlg = myLogisticRegression(learning_rate=1e-8, opt_method = 'SGD', num_epochs = 15, size_batch = 40)
mlg.fit(X_train,y_train,n_iterations = 1000)


# In[23]:


mlg.score(X_test,y_test)


# Performance Report: 
# 
# With a score of about 0.6264, I would expect this model to correctly recognize Kuzushiji characters about 62% of the time. 

# ---

# ## Task 3: Princle Component Analysis 

# In[24]:


import numpy as np

class myPCA():
    '''
    Goals:
    - standardize the range of the continuous initial variables
    - reduce the number of variables of a data set, while preserving as much information as possible
    '''
    
    
    def __init__(self, n_components = 2):
        '"initiate n-components"'
        self.n_c = n_components
    
    def fit(self,X):
        '''
        Step 1
        - understand how the variables of the input data set are varying from the mean with respect to each other
        using covariance matrix
        Step 2
        - compute eigenvectors and eigenvalues from the covariance matrix in order to determine the principal components
        Step 3
        - describe your data in terms of new desired variables (principal components) 
        '''
        cov_mat = np.cov(X.T) # covariance matrix, the input matrix to this function does not need to be centered
        eig_val, eig_vec = np.linalg.eigh(cov_mat) #eigen-values and orthogonal eigen-vectors --ascending order
        eig_val = np.flip(eig_val) # reverse the order --descending
        eig_vec = np.flip(eig_vec,axis=1) # reverse the order
        self.eig_values = eig_val[:self.n_c] # select the top eigen-vals
        self.principle_components = eig_vec[:,:self.n_c] # select the top eigen-vecs
        self.variance_ratio = self.eig_values/eig_val.sum() # variance explained by each PC
    
    def transform(self,X):
        '"Reorient the data from the original axes to the ones represented by the principal components"'
        return np.matmul(X-X.mean(axis = 0),self.principle_components) #project the data (centered) on PCs


# In[25]:


Data = [kmnist_train1, kmnist_test1]
data1 = pd.concat(Data)
X = data1.drop('label', axis = 1).to_numpy()  # create combined X and y data sets for unsupervised learning tasks
y = data1[['label']].to_numpy().ravel()   # ravel works like flatten


# In[26]:


pca = myPCA(n_components = 15)
pca.fit(X)
X_pca = pca.transform(X)
X_pca      # principle components 


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

pca1 = myPCA(n_components = 15)
pca1.fit(X)                   
X_pca1 = pca1.transform(X)

figure = plt.figure(dpi=100)
plt.scatter(X_pca1[:, 0], X_pca1[:, 1],c=y, s=15, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('tab10', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# ---

# ## Task 4: Scikit-Learn

# **Decision Tree:**
#     
# Decision Tree Learning in classification uses observations of an object to identify its label. Intuitively, decision trees can be thought of as setting the threshold for different features of an object by use of multiple if-else conditions. Thus, forming a flow-chart or decision tree structure with each branch representing a distinguishing characterstic. 
# 
# Mathematically, the decision tree model uses Gini impurity to measure how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. 
# This can be computed by summing the probability $p_{i}$ of an object with label ${\displaystyle i}$ being chosen times the probability $\sum_{k\neq i} p_k = {1-p_i}$ of making a mistake
# 
# To compute Gini impurity for a set of objects with $J$ classes, suppose ${ i\in \{1,2,...,J\}}$, and let $p_{i}$ be the fraction of objects labeled with class $i$ in the set.
# 
# $${\displaystyle \operatorname {I} _{G}(p)=\sum _{i=1}^{J}\left(p_{i}\sum _{k\neq i}p_{k}\right)=\sum _{i=1}^{J}p_{i}(1-p_{i})=\sum _{i=1}^{J}(p_{i}-{p_{i}}^{2})=\sum _{i=1}^{J}p_{i}-\sum _{i=1}^{J}{p_{i}}^{2}=
# 1-\sum _{i=1}^{J}{p_{i}}^{2}}{\displaystyle \operatorname {I} _{G}(p)=\sum _{i=1}^{J}\left(p_{i}\sum _{k\neq i}p_{k}\right)=\sum _{i=1}^{J}p_{i}(1-p_{i})=\sum _{i=1}^{J}(p_{i}-{p_{i}}^{2})=\sum _{i=1}^{J}p_{i}-\sum _{i=1}^{J}{p_{i}}^{2}=1-\sum _{i=1}^{J}{p_{i}}^{2}}$$
# 

# In[28]:


from sklearn import tree

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train) # compute Gini impurity
dtc.score(X_test,y_test)


# With a score of 0.6346, we can see that the decision tree learning method for classification performed about 1% better than our Logistic Regression model. 

# **K-means Clustering**
# 
# The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
# 
# 1. The centroids of the K clusters, which can be used to label new data
# 
# 2. Labels for the training data (each data point is assigned to a single cluster)
# 
# Mathematically, given a set of observations $(x^{(1)}, x^{(2)}, ..., x^{(n)})$, where each observation is a p-dimensional real vector, k-means clustering aims to partition the $n$ samples into $K (\leq n)$ sets $S = {S_1, S_2, ..., S_K}$ so as to minimize the within-cluster sum of squares (i.e. variance). Formally, the objective is to find the best parition of groups such that minimize the "loss function" of $S$
# 
# $$\min_{S}\sum_{i=1}^{K}\sum_{x\in S_{i}}\|x-\mu_{i}\|^{2}$$
# 
# where $\mu_{i}$ is the mean of points in $S_i$.
# 
# 

# In[29]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
y_km = kmeans.fit_predict(X)
y_km


# In[30]:


from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X)


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
fig, (ax1, ax2) = plt.subplots(1, 2,dpi=150)

fig1 = ax1.scatter(X_pca2[:, 0], X_pca2[:, 1],c=y_km, s=15, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('Set1', 3))
fig2 = ax2.scatter(X_pca2[:, 0], X_pca2[:, 1],c=y, s=15, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('Accent', 3))
ax1.set_title('K-means Clustering')
legend1 = ax1.legend(*fig1.legend_elements(), loc="best", title="Classes")
ax1.add_artist(legend1)
ax2.set_title('True Labels')
legend2 = ax2.legend(*fig2.legend_elements(), loc="best", title="Classes")
ax2.add_artist(legend2)


# In[32]:


from sklearn import metrics
metrics.adjusted_rand_score(y_km, y)


# ---

# ## Task 5: Tensorflow

# In[33]:


pip install tensorflow


# In[34]:


import tensorflow
print(tensorflow.__version__)


# In[35]:


from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X1 = X.astype('float32')
# encode strings to integer
y1 = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)
# determine the number of input features
n_features = X_train1.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train1, y_train1, epochs=150, batch_size=32, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test1, y_test1, verbose=0)
print('Test Accuracy: %.3f' % acc)
# make a prediction
yhat = model.predict(X_test)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))


# With a test accuracy of about 76%, we can see that this tensorflow model was more effective than both the Decision tree model and the Logistic Regression model. 
