# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:20:18 2019

@author: 
Kishan Patel
NETId: Kpp73
Student ID: 31488298
"""
import sys;
import csv;
import statistics as sts;
import random as rnd;
import numpy as np;
from sklearn import svm
from sklearn import linear_model;
import os;

csv.register_dialect('spc', delimiter = ' ', escapechar = '\n');

# Load the Data and Label Files [2d matrices]
def readfile(fname):
    with open(fname, 'r') as f:
        reader = csv.reader(f, dialect = 'spc');
        data = [];
        for r in reader:
            dt = [];
            for c in r:
                if c != '':
                    dt.append(int(c));
            data.append(dt);
        return data

# Get the training labels
def gettrainlabels(fname):
    Y = readfile(fname);
    L = [Y[i][0] for i in range(0, len(Y))]; # First column in the TrueClass file are the labels
    return L


# Output File for the testclass prediction and features
def writetestpred(testpred, indi, X_train15, X_val15, X_test15, acr):
    T = testpred;
    X = X_train15;
    XV = X_val15;
    XT = X_test15;
    fnum = len(indi);

    if os.path.exists('Kishan_Testclass_Prediction.txt'):
        os.remove('Kishan_Testclass_Prediction.txt');
    elif os.path.exists('Kishan_Features_Files.txt'):
        os.remove('Kishan_Features_Files.txt');

    pred = open('Kishan_Testclass_Prediction.txt', 'w+');
    #pred.write('Kishan Patel\nNetID: Kpp73\nStudentID: 31488298')
    #pred.write('\nNumber of features used: {}\n'.format(fnum));
    #pred.write('\nColumns used in prediction: {}\n'.format(indi));
    #pred.write('\nThe accuracy of this model is {}%\n'.format(acr));
    #pred.write('Test Data Prediction---------------------------------------------------------------------------------------\n');
    #pred.write('\n');
    for i in range(0, len(T)):
        pred.write('{} {}'.format(T[i], i));
        #print('{} {}'.format(T[i], i));
        pred.write('\n');
        #print('\n');
    pred.close();

    fea = open('Kishan_Features_Files.txt', 'w+');
    fea.write('Kishan Patel\nNetID: Kpp73\nStudentID: 31488298')
    fea.write('\nNumber of features used: {}\n'.format(fnum));
    #fea.write('\nColumns used in prediction: {}\n'.format(indi));
    fea.write('\nThe accuracy of this model is {}%\n'.format(acr));
    fea.write('\nFeatures Used in Model:---------------------------------------------------------------------------------------------\n');
    fea.write('\n');
    for i in range(0, len(indi)):
        fea.write('{} '.format(indi[i]));
    #for i in range(0, len(XT)):
        #fea.write('{}'.format(XT[i]));
        #fea.write('\n');
    #fea.write('Train Features-----------------------------------------------------------------------------------------------\n');
    #fea.write('\n');
    #for i in range(0, len(X)):
        #fea.write('{}'.format(X[i]));
        #fea.write('\n');
    #fea.write('Validation Features------------------------------------------------------------------------------------------\n');
    #fea.write('\n');
    #for i in range(0, len(XV)):
        #fea.write('{}'.format(XV[i]));
        #fea.write('\n');
    fea.close();

# Feature Selection using Chi-Square
def chi_sqr(X, y, topf):
    rows = len(X);
    cols = len(X[0]);
    O = [];
    for j in range(0, cols):
        ctable = [[1,1],[1,1],[1,1]];
        for i in range(0, rows):
            if y[i] == 0:
                if X[i][j] == 0:
                    ctable[0][0] += 1
                elif X[i][j] == 1:
                    ctable[1][0] += 1
                elif X[i][j] == 2:
                    ctable[2][0] += 1
            elif y[i] == 1:
                if X[i][j] == 0:
                    ctable[0][1] += 1
                elif X[i][j] == 1:
                    ctable[1][1] += 1
                elif X[i][j] == 2:
                    ctable[2][1] += 1
        ctot = [sum(c) for c in ctable];
        rtot = [sum(c) for c in zip(*ctable)];
        tot = sum(ctot);
        exp = [[(r*c)/tot for r in rtot] for c in ctot];
        sqr = [[((ctable[i][j] - exp[i][j])**2)/exp[i][j] for j in range(0,len(exp[0]))] for i in range(0,len(exp))];
        chi2 = sum([sum(x) for x in zip(*sqr)])
        O.append(chi2);
    ind = sorted(range(len(O)), key=O.__getitem__, reverse=True); # Sorts features from best to worse, in respect to chi2 values
    indi= ind[:topf]; # Returns the top 15 features column-indicies
    return indi

# Extracting the Selected Features
def fextract(X, c):
    EF = [];
    col = list(zip(*X));
    for i in c:
        EF.append(col[i]);
    EF = list(zip(*EF));
    return EF

def frankselect(file_train_data, file_train_labels, file_test_data):
    X = readfile(file_train_data);
    Y = gettrainlabels(file_train_labels);
	
    # Split into train/validate (Cross-Validation)
    Z = list(zip(Y,X));
    rnd.shuffle(Z);
    Y,X = zip(*Z);

    X = np.array(X); # Convert to Array to save space
    Y = np.array(Y); # Convert to Array to save space
	
    r1 = int(len(X) * 0.75);
    X_train = X[:r1];
    X_val = X[r1:];
    Y_train = Y[:r1];
    Y_val = Y[r1:];

    # Feature Selection via Chi-Square [Top 15]
    indi = chi_sqr(X_train, Y_train, 15)

    # Extract Selected Features
    X_train15 = fextract(X_train, indi);
    X_train15 =np.array(X_train15); # Convert to Array to save space

    X_val15 = fextract(X_val, indi);
    X_val15 = np.array(X_val15); # Convert to Array to save space

    # Read and Extract Test Data
    X_test = readfile(file_test_data);
    X_test = np.array(X_test); # Convert to Array to save space
    X_test15 = fextract(X_test, indi);
    X_test15 = np.array(X_test15); # Convert to Array to save space

    fnum = len(indi);
    print('\nNumber of features used: {}'.format(fnum))
    print('\nColumns used in prediction: {}'.format(indi))
	
    return X_train15, X_val15, X_test15, Y_train, Y_val, indi

# Create and Execute Model
def traintest(X_train15, X_val15, X_test15, Y_train, Y_val):
    linsvc = svm.LinearSVC(C=100.0,max_iter=100000);
    svc = svm.SVC(C=100.0,kernel='linear',tol=0.0001);
    logreg = linear_model.LogisticRegression(C=100.0,max_iter=100000);

    # Train Model
    t1 = linsvc.fit(X_train15,Y_train);
    t2 = svc.fit(X_train15,Y_train);
    t3 = logreg.fit(X_train15,Y_train);

    # Validate Model [Using 'Max-Vote' Ensemble method]
    tv1 = linsvc.predict(X_val15);
    tv2 = svc.predict(X_val15);
    tv3 = logreg.predict(X_val15);
    val_pred = np.array([]);
    for i in range(0,len(X_val15)):
        val_pred = np.append(val_pred, sts.mode([tv1[i],tv2[i],tv3[i]]))
    sc = 0;
    for i in range(0,len(Y_val)):
        if (val_pred[i] == Y_val[i]):
            sc += 1;
    acr = (sc/len(Y_val))*100;
    print('\nThe accuracy of this model is {}%'.format(acr));

    # Test with TestData
    tt1 = linsvc.predict(X_test15);
    tt2 = svc.predict(X_test15);
    tt3 = logreg.predict(X_test15);
    test_pred = np.array([]);
    for i in range(0,len(X_test15)):
        test_pred = np.append(val_pred, sts.mode([tt1[i],tt2[i],tt3[i]]))
    return test_pred, acr

if __name__ == '__main__':
    trainfile = sys.argv[1];
    tlabelfile = sys.argv[2];
    testfile = sys.argv[3];

    # Feature Select and Extraction
    X_train15, X_val15, X_test15, Y_train, Y_val, indi = frankselect(trainfile, tlabelfile, testfile);
    # Train and Test Model
    test_pred, acr = traintest(X_train15, X_val15, X_test15, Y_train, Y_val);
    # Write Output File
    writetestpred(test_pred, indi, X_train15, X_val15, X_test15, acr);
