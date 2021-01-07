import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def getCov(X, uk):
    cov = np.zeros((X.shape[1],X.shape[1]))
    for i in range(X.shape[0]):
        cov += (X[i,:] - uk)[:,None] @ (X[i,:] - uk)[:,None].T
    return cov

# predicting
def predict(X, cov, u, K):
    criterion = np.zeros((X.shape[0], K))
    criterion[:,0] = X @ np.linalg.inv(cov) @ u[0] - 0.5*u[0][:,None].T @ np.linalg.inv(cov) @ u[0] + np.log(1/3)
    criterion[:,1] = X @ np.linalg.inv(cov) @ u[1] - 0.5*u[1][:,None].T @ np.linalg.inv(cov) @ u[1] + np.log(1/3)
    criterion[:,2] = X @ np.linalg.inv(cov) @ u[2] - 0.5*u[2][:,None].T @ np.linalg.inv(cov) @ u[2] + np.log(1/3)
    pred=[]
    for i in range(criterion.shape[0]):
        pred.append(np.argmax(criterion[i,:]))
    pred = np.array(pred)
    return pred

if __name__ == '__main__':
    np.random.seed(0)
    
    train_data = pd.read_csv(r'data/train.csv', header=None).values
    train_data = np.delete(train_data, 0, axis = 0)
    train_data = train_data.astype(np.int)
    
    test_data = pd.read_csv(r'data/test.csv', header=None).values
    test_data = np.delete(test_data, 0, axis = 0)
    test_data = test_data.astype(np.int)
    
    #training data preprocessing
    train_mask = []
    for i in range(0, train_data.shape[0], 1):
        if train_data[i][0] == 0 or train_data[i][0] == 1 or train_data[i][0] == 2:
            train_mask.append(i)

    train_mask = np.array(train_mask)
    train_data = train_data[train_mask,:]
    train_data_x = train_data[:,1:]
    train_data_y = train_data[:,0]
    train_data_x = train_data_x/255 #normalization input data
    #add noise to data to avoid correlation between predictors
    train_data_x = train_data_x + 0.0001*np.random.rand(train_data_x.shape[0], train_data_x.shape[1])

    #testing data preprocessing
    test_mask = []
    for i in range(0, test_data.shape[0], 1):
        if test_data[i][0] == 0 or test_data[i][0] == 1 or test_data[i][0] == 2:
            test_mask.append(i)

    test_mask = np.array(test_mask)
    test_data = test_data[test_mask,:]
    test_data_x = test_data[:,1:]
    test_data_y = test_data[:,0]
    test_data_x = test_data_x/255 #normalization input data
    #add noise to data to avoid correlation between predictors
    test_data_x = test_data_x + 0.0001*np.random.rand(test_data_x.shape[0], test_data_x.shape[1])
    
    u = [np.mean(train_data_x[np.where(train_data_y==i)[0]], axis=0) for i in [0,1,2]]
    K = 3
    cov0 = getCov(train_data_x[np.where(train_data_y==0)[0],:], u[0])
    cov1 = getCov(train_data_x[np.where(train_data_y==1)[0],:], u[1])
    cov2 = getCov(train_data_x[np.where(train_data_y==2)[0],:], u[2])
    cov = (1/(train_data_x.shape[0]-K))*(cov0 + cov1 + cov2)

    test_preds = predict(test_data_x, cov, u, K)
    # Plot confusion matrix
    confusion_matrix = pd.crosstab(pd.Series(test_data_y, name='True'), pd.Series(test_preds, name='Prediction'))
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()