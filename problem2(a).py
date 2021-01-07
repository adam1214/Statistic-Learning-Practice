import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.0001, num_iter=100000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __getGD(self, X, y):
        # gradient initialization
        grad = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            numer = (2*y[i]-1) * X[i,:]
            denom = 1+np.exp((2*y[i]-1)* (self.beta.T @ X[i,:]))
            grad += (numer/denom)
        grad = -grad
        
        return grad
    
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # estimator initialization
        self.beta = np.zeros(X.shape[1]) #785*1

        for i in range(self.num_iter):   
            print('k = %d' % i)         
            gradient = self.__getGD(X, y)
            self.beta = self.beta - self.lr*gradient
            
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        prob = (np.exp(X @ self.beta) / (1+np.exp(X @ self.beta)))

        return prob
    
    def predict(self, X, threshold):
        prob = self.predict_prob(X)
        test_preds = np.zeros(len(prob))
        for i in range(0, test_preds.shape[0], 1):
            if prob[i] >= threshold:
                test_preds[i] = 1
        return test_preds

if __name__ == '__main__':
    train_data = pd.read_csv(r'data/train.csv', header=None).values
    train_data = np.delete(train_data, 0, axis = 0)
    train_data = train_data.astype(np.int)
    
    test_data = pd.read_csv(r'data/test.csv', header=None).values
    test_data = np.delete(test_data, 0, axis = 0)
    test_data = test_data.astype(np.int)
    
    #training data preprocessing
    train_mask = []
    for i in range(0, train_data.shape[0], 1):
        if train_data[i][0] == 1 or train_data[i][0] == 2:
            train_mask.append(i)

    train_mask = np.array(train_mask)
    train_data = train_data[train_mask,:]
    train_data_x = train_data[:,1:]
    train_data_y = train_data[:,0]
    for i in range(0, train_data_y.shape[0], 1): # 0, 1 classification problem
        if train_data_y[i] == 1:
            train_data_y[i] = 0
        else:
            train_data_y[i] = 1
    train_data_x = train_data_x/255 #normalization input data

    #testing data preprocessing
    test_mask = []
    for i in range(0, test_data.shape[0], 1):
        if test_data[i][0] == 1 or test_data[i][0] == 2:
            test_mask.append(i)

    test_mask = np.array(test_mask)
    test_data = test_data[test_mask,:]
    test_data_x = test_data[:,1:]
    test_data_y = test_data[:,0]
    for i in range(0, test_data_y.shape[0], 1): # 0, 1 classification problem
        if test_data_y[i] == 1:
            test_data_y[i] = 0
        else:
            test_data_y[i] = 1
    test_data_x = test_data_x/255 #normalization input data
    
    learning_rate = 0.001
    num_iter = 11
    model = LogisticRegression(learning_rate, num_iter)
    model.fit(train_data_x, train_data_y)
    test_preds = model.predict(test_data_x, 0.5)
    print('step size = %f' % learning_rate)
    print('所有beta的初值皆為0')
    print('beta總共更新%d次' % (num_iter-1))

    # Plot confusion matrix
    confusion_matrix = pd.crosstab(pd.Series(test_data_y, name='True'), pd.Series(test_preds, name='Prediction'))
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()