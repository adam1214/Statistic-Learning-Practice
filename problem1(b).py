import csv
import numpy as np
from scipy import stats
from itertools import combinations 

def get_Coefficient(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

def get_Std_err(X, Y, Y_pred, n, p): # HW1 1.(a)
    sigma_hat = (1/(n-p-1))*sum((Y-Y_pred)**2)
    SE = []
    tmp = np.linalg.inv(X.T @ X)*sigma_hat
    for j in range(p+1):
        SE.append(np.sqrt(tmp[j,j]))
    return np.array(SE)

def RSE(Y, Y_pred, n, p):
    RSS = sum((Y-Y_pred)**2)
    return np.sqrt((1/(n-p-1))*RSS)

def Rsquare(Y, Y_pred):
    RSS = sum((Y-Y_pred)**2)
    TSS = sum((Y-np.mean(Y))**2)
    return 1-(RSS/TSS)

if __name__ == '__main__':
    comb = list(combinations([1, 2, 3, 4, 5, 6], 2))
    feature_list = ['seafood', 'meat', 'offals', 'spices', 'vegetables', 'obesity']
    X_list = []
    RSS = []
    Coefficient_list = []
    Y_pred_list = []
    for c in comb:
        #print(c[0], c[1])
        X = np.zeros( (160, 3) )
        Y = np.zeros( (160, 1) )
        line_num = 0
        csv_file=open('data/covid-19.csv')
        csv_reader_lines = csv.reader(csv_file)
        for one_line in csv_reader_lines:
            if line_num != 0:
                X[line_num-1][0] = 1
                for i in range(1, 7, 1):
                    if i == c[0]:
                        X[line_num-1][1] = one_line[i]
                    elif i == c[1]:
                        X[line_num-1][2] = one_line[i]
                Y[line_num-1][0] = one_line[7]
            line_num += 1
        X_list.append(X)
        Coefficient = get_Coefficient(X, Y)
        Coefficient_list.append(Coefficient)
        Y_pred = X @ Coefficient
        Y_pred_list.append(Y_pred)
        RSS.append(sum((Y-Y_pred)**2))
    #print(RSS)
    target_index = np.argmin(RSS)
    X = X_list[target_index]
    Y_pred = Y_pred_list[target_index]
    target_predictor = comb[target_index]
    print('feature 1 is %s (X%d)' %(feature_list[target_predictor[0]-1], target_predictor[0]))
    print('feature 2 is %s (X%d)' %(feature_list[target_predictor[1]-1], target_predictor[1]))
    print('############################################################')
    Coefficient = get_Coefficient(X, Y)
    print('Coefficient of X%d:\t%f' % (target_predictor[0], Coefficient[1]))
    print('Coefficient of X%d:\t%f' % (target_predictor[1], Coefficient[2]))
    print('############################################################')
    Std_err = get_Std_err(X, Y, Y_pred, n = 160, p = 2)
    print('Std. error of X%d:\t%f' % (target_predictor[0], Std_err[1]))
    print('Std. error of X%d:\t%f' % (target_predictor[1], Std_err[2]))
    print('############################################################')
    t = (Coefficient.T)/Std_err
    print('t-statistic of X%d:\t%f' % (target_predictor[0], t[0][1]))
    print('t-statistic of X%d:\t%f' % (target_predictor[1], t[0][2]))
    print('############################################################')
    p_values = []
    for t_score in t[0]:
        p_values.append(2*(1-stats.t.cdf(np.abs(t_score), (160-2-1))))
    print('p-value of X%d:\t%f' % (target_predictor[0], p_values[1]))
    print('p-value of X%d:\t%f' % (target_predictor[1], p_values[2]))
    print('############################################################')
    print('RSE:', float(RSE(Y, Y_pred, n = 160, p = 2)))
    print('R^2:', float(Rsquare(Y, Y_pred)))
    