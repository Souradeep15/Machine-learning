# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:29:34 2019

@author: Kishan Patel
"""
import sys;
import random as rnd;
import math;

def bestginisplit(X_train, Y_train, col):
    cv = {}; indi = []; r = 0; m = 0;
    ls = 1; lp = 0; bestsplit = -1; bestgini = 10000;
    for i in range(0, len(X_train), 1):
        if(Y_train.get(i) != None):
            cv[i] = X_train[i][col]
            indi.append(i); r += 1;
            if(Y_train[i] == 0):
                m += 1
    si = sorted(indi, key = cv.__getitem__);
    rs = r - 1; rp = m;

    if(Y_train[si[0]] == 0):
        lp += 1; rp -= 1;

    # Finding Best Gini Index
    for i in range(1, len(si), 1):
        split = (cv[si[1]] + cv[si[i-1]])/2;
        gini = (ls/r) * (lp/ls) * (1-(lp/ls)) + (rs/r) * (rp/rs) * (1-(rp/rs));
        if(gini < bestgini):
            bestgini = gini;
            bestsplit = split;
        if(Y_train[si[i]] == 0):
            lp += 1; rp -= 1;
        ls += 1; rs -= 1;

    return(bestsplit, bestgini);

# Read The Data
datafile = sys.argv[1];
labelfile = sys.argv[2];
f = open(datafile);
l = f.readline();
data = [];

while (l != ''):
    a = l.split();
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]));
    data.append(l2);
    l = f.readline();

rows = len(data);
cols = len(data[0]);
f.close();

# Read the Training Labels
trainlabels = {};
n = [];
f = open(labelfile);
n = [];
n.append(0);
n.append(0);
l = f.readline();

while (l != ''):
    a = l.split();
    trainlabels[int(a[1])] =  int(a[0]);
    n[int(a[0])] += 1;
    l = f.readline()
f.close();

#Bagged Decision Stump
tboots = 100;
testpred = {};
for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):
        testpred[i] = 0;
for k in range(0, tboots, 1):
    x = 0;
    bdata = []; btlabels = {};
    while(x < len(data)):
        rn = rnd.randint(0, rows - 1);
        if(trainlabels.get(rn) != None):
            bdata.append(data[rn]);
            btlabels[x] = trainlabels[rn];
            x += 1;
    bestsplit = -1; bestcol = -1; bestgini = 100000;
    for i in range(0, cols, 1):
        [split, gini] = bestginisplit(bdata, btlabels, i);
        if(gini < bestgini):
            bestgini = gini;
            bestsplit = split;
            bestcol = i;
    m = 0; p = 0;
    for j in range(0, rows, 1):
        if(trainlabels.get(j) != None):
            if(data[j][bestcol] < bestsplit):
                if(trainlabels[j] == 0):
                    m += 1;
                else:
                    p += 1;
    if(m > p):
        l = -1; r = 1;
    else:
        l = 1; r = -1;

    for j in range(0, rows, 1):
        if(trainlabels.get(j) == None):
            if(data[j][bestcol] < bestsplit):
                testpred[j] += l;
            else:
                testpred[j] += r;
# Prediction
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        if(testpred[i] > 0):
            print("1", i);
        else:
            print("0", i);




