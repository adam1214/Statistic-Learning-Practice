import csv
import numpy as np
from scipy import stats

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
    csv_file=open('data/covid-19.csv')
    csv_reader_lines = csv.reader(csv_file)

    X = np.zeros( (160, 7) )
    Y = np.zeros( (160, 1) )
    line_num = 0
    for one_line in csv_reader_lines:
        if line_num != 0:
            X[line_num-1][0] = 1
            for i in range(1, 7, 1):
                X[line_num-1][i] = one_line[i]
            Y[line_num-1][0] = one_line[7]
        line_num += 1

    Coefficient = get_Coefficient(X, Y)
    print('Coefficient of intercept:\t%f' % Coefficient[0])
    for i in range(1, 7, 1):
        print('Coefficient of X%d:\t\t%f' % (i, Coefficient[i]))
    print('############################################################')
    Y_pred = X @ Coefficient
    Std_err = get_Std_err(X, Y, Y_pred, n = 160, p = 6)
    print('Std. error of intercept:\t%f' % Std_err[0])
    for i in range(1, 7, 1):
        print('Std. error of X%d:\t\t%f' % (i, Std_err[i]))
    print('############################################################')
    t = (Coefficient.T)/Std_err
    print('t-statistic of intercept:\t%f' % t[0][0])
    for i in range(1, 7, 1):
        print('t-statistic of X%d:\t\t%f' % (i, t[0][i]))
    print('############################################################')
    #p_values =[2*(1-stats.t.cdf(np.abs(i),(414-6))) for i in t]
    p_values = []
    for t_score in t[0]:
        p_values.append(2*(1-stats.t.cdf(np.abs(t_score), (160-6-1))))
    print('p-value of intercept:\t%f' % p_values[0])
    for i in range(1, 7, 1):
        print('p-value of X%d:\t\t%f' % (i, p_values[i]))
    print('############################################################')
    print('RSE:', float(RSE(Y, Y_pred, n = 160, p = 6)))
    print('R^2:', float(Rsquare(Y, Y_pred)))