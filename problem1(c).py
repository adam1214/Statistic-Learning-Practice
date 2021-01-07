import csv
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import math

def get_Coefficient(X, Y, lamda):
    return np.linalg.inv((X.T @ X) + lamda * np.eye(2)) @ X.T @ Y

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
    lamda_list = [10**-5, 10**-4.5, 10**-4, 10**-3.5, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**-0.5, 10**0, 10**0.5, 10**1, 10**1.5, 10**2, 10**2.5, 10**3]
    CV_list = []
    lamda_feature_coeff_dict = {}
    RSE_list = []
    Rsquare_list = []

    X_fold1 = np.zeros( (32, 7) )
    X_fold2 = np.zeros( (32, 7) )
    X_fold3 = np.zeros( (32, 7) )
    X_fold4 = np.zeros( (32, 7) )
    X_fold5 = np.zeros( (32, 7) )
    
    Y_fold1 = np.zeros( (32, 1) )
    Y_fold2 = np.zeros( (32, 1) )
    Y_fold3 = np.zeros( (32, 1) )
    Y_fold4 = np.zeros( (32, 1) )
    Y_fold5 = np.zeros( (32, 1) )
    
    line_num = 0
    csv_file=open('data/covid-19.csv')
    csv_reader_lines = csv.reader(csv_file)
    for one_line in csv_reader_lines:
        if line_num != 0:
            for i in range(1, 7, 1):
                if line_num >= 1 and line_num < 33: #1~32
                    X_fold1[line_num-1][i] = one_line[i]
                    Y_fold1[line_num-1][0] = one_line[7]
                elif line_num >= 33 and line_num < 65: #33~64
                    X_fold2[line_num-1-32][i] = one_line[i]
                    Y_fold2[line_num-1-32][0] = one_line[7]
                elif line_num >= 65 and line_num < 97: #65~96
                    X_fold3[line_num-1-64][i] = one_line[i]
                    Y_fold3[line_num-1-64][0] = one_line[7]
                elif line_num >= 97 and line_num < 129: #97~128
                    X_fold4[line_num-1-96][i] = one_line[i]
                    Y_fold4[line_num-1-96][0] = one_line[7]
                elif line_num >= 129 and line_num < 161: #129~160
                    X_fold5[line_num-1-128][i] = one_line[i]
                    Y_fold5[line_num-1-128][0] = one_line[7]
        line_num += 1

    comb = list(combinations([1, 2, 3, 4, 5, 6], 2))
    feature_list = ['seafood', 'meat', 'offals', 'spices', 'vegetables', 'obesity']
    for lamda in lamda_list:
        print('lamda = %f:' % lamda)
        Centered_X_list = []
        RSS = []
        Coefficient_list = []
        Y_pred_list = []
        for c in comb:
            #print(c[0], c[1])
            X = np.zeros( (160, 2) )
            Y = np.zeros( (160, 1) )
            line_num = 0
            csv_file=open('data/covid-19.csv')
            csv_reader_lines = csv.reader(csv_file)
            for one_line in csv_reader_lines:
                if line_num != 0:
                    for i in range(1, 7, 1):
                        if i == c[0]:
                            X[line_num-1][0] = one_line[i]
                        elif i == c[1]:
                            X[line_num-1][1] = one_line[i]
                    Y[line_num-1][0] = one_line[7]
                line_num += 1
            Centered_Y = Y - np.mean(Y) #centered Y
            Centered_X = X - np.mean(X, 0) #centered X
            Centered_X_list.append(Centered_X)
            Coefficient = get_Coefficient(Centered_X, Centered_Y, lamda)
            Coefficient_list.append(Coefficient)
            Y_pred = Centered_X @ Coefficient
            Y_pred_list.append(Y_pred)
            RSS.append(sum((Centered_Y-Y_pred)**2))
        #print(RSS)
        target_index = np.argmin(RSS)
        Centered_X = Centered_X_list[target_index]
        Y_pred = Y_pred_list[target_index]
        target_predictor = comb[target_index]
        print('feature 1 is %s (X%d)' %(feature_list[target_predictor[0]-1], target_predictor[0]))
        print('feature 2 is %s (X%d)' %(feature_list[target_predictor[1]-1], target_predictor[1]))
        Coefficient = get_Coefficient(Centered_X, Centered_Y, lamda)
        print('Coefficient of X%d:\t%f' % (target_predictor[0], Coefficient[0]))
        print('Coefficient of X%d:\t%f' % (target_predictor[1], Coefficient[1]))
        lamda_feature_coeff_dict[lamda] = [(feature_list[target_predictor[0]-1]+'(X'+str(target_predictor[0])+')', feature_list[target_predictor[1]-1]+ '(X'+str(target_predictor[1])+')'), (Coefficient[0], Coefficient[1])]
        RSE_list.append(float(RSE(Centered_Y, Y_pred, n = 160, p = 2)))
        Rsquare_list.append(float(Rsquare(Centered_Y, Y_pred)))
        
        #Start CV
        #get 2 predictors from data
        subX_fold1 = np.zeros( (32, 2) )
        subX_fold2 = np.zeros( (32, 2) )
        subX_fold3 = np.zeros( (32, 2) )
        subX_fold4 = np.zeros( (32, 2) )
        subX_fold5 = np.zeros( (32, 2) )
        for i in range(0, 32, 1):
            subX_fold1[i][0] = X_fold1[i][target_predictor[0]]
            subX_fold1[i][1] = X_fold1[i][target_predictor[1]]
            
            subX_fold2[i][0] = X_fold2[i][target_predictor[0]]
            subX_fold2[i][1] = X_fold2[i][target_predictor[1]]
            
            subX_fold3[i][0] = X_fold3[i][target_predictor[0]]
            subX_fold3[i][1] = X_fold3[i][target_predictor[1]]
            
            subX_fold4[i][0] = X_fold4[i][target_predictor[0]]
            subX_fold4[i][1] = X_fold4[i][target_predictor[1]]
            
            subX_fold5[i][0] = X_fold5[i][target_predictor[0]]
            subX_fold5[i][1] = X_fold5[i][target_predictor[1]]
        
        MSE_total = 0
        for j in range(1, 6, 1):
            if j == 1:
                train_x = np.concatenate((subX_fold2,subX_fold3,subX_fold4,subX_fold5), axis=0)
                train_y = np.concatenate((Y_fold2,Y_fold3,Y_fold4,Y_fold5), axis=0)
                test_x = subX_fold1
                test_y = Y_fold1
            elif j == 2:
                train_x = np.concatenate((subX_fold1,subX_fold3,subX_fold4,subX_fold5), axis=0)
                train_y = np.concatenate((Y_fold1,Y_fold3,Y_fold4,Y_fold5), axis=0)
                test_x = subX_fold2
                test_y = Y_fold2
            elif j == 3:
                train_x = np.concatenate((subX_fold1,subX_fold2,subX_fold4,subX_fold5), axis=0)
                train_y = np.concatenate((Y_fold1,Y_fold2,Y_fold4,Y_fold5), axis=0)
                test_x = subX_fold3
                test_y = Y_fold3
            elif j == 4:
                train_x = np.concatenate((subX_fold1,subX_fold2,subX_fold3,subX_fold5), axis=0)
                train_y = np.concatenate((Y_fold1,Y_fold2,Y_fold3,Y_fold5), axis=0)
                test_x = subX_fold4
                test_y = Y_fold4
            elif j == 5:
                train_x = np.concatenate((subX_fold1,subX_fold2,subX_fold3,subX_fold4), axis=0)
                train_y = np.concatenate((Y_fold1,Y_fold2,Y_fold3,Y_fold4), axis=0)
                test_x = subX_fold5
                test_y = Y_fold5
            train_y = train_y - np.mean(train_y) #centered Y
            train_x = train_x - np.mean(train_x, 0) #centered X
            Coeff = get_Coefficient(train_x, train_y, lamda)
            Y_pred = (test_x - np.mean(test_x, 0)) @ Coeff
            MSE_total = MSE_total + ( (sum(((test_y - np.mean(test_y))-Y_pred)**2))/32 )
        CV_list.append(float(MSE_total/5))
        print('############################################################')
    
    min_CVerr_index = CV_list.index(min(CV_list))# 最小值的索引
    print('當lamda = %f時，Cross-Validation Error最小' % lamda_list[min_CVerr_index])
    print('Feature1為%s' % lamda_feature_coeff_dict[lamda_list[min_CVerr_index]][0][0])
    print('Feature2為%s' % lamda_feature_coeff_dict[lamda_list[min_CVerr_index]][0][1])
    print('Coefficient of Feature1:%f' % lamda_feature_coeff_dict[lamda_list[min_CVerr_index]][1][0])
    print('Coefficient of Feature2:%f' % lamda_feature_coeff_dict[lamda_list[min_CVerr_index]][1][1])
    print('RSE:', RSE_list[min_CVerr_index])
    print('R^2:', Rsquare_list[min_CVerr_index])
    #print(lamda_feature_coeff_dict[lamda_list[min_CVerr_index]])
    for i in range(0, len(lamda_list), 1): # lamda in log-scale
        lamda_list[i] = math.log(lamda_list[i], 10)
    plt.plot(lamda_list, CV_list)
    plt.xlabel('lamda in log-scale')
    plt.ylabel('Cross-Validation Error')
    plt.show()
    