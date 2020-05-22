# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:05:14 2019

@author: Kishan Patel
"""
import sys
import numpy as py

dataf = sys.argv[1];
f = open(dataf);
data = [];
i = 0;
l = f.readline();

# READ THE DATA
while(l != ''):
    a = l.split();
    l2 = [];
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()

rows = len(data);
cols = len(data[0]);
f.close();

# READ THE LABELS
lfile = sys.argv[2];
f = open(lfile);
trainlabels = {};
n = [];
n.append(0);
n.append(0);
l = f.readline();
while(l != ''):
    a = l.split();
    trainlabels[int(a[1])] = int(a[0]);
    l = f.readline();
    n[int(a[0])] += 1

# COMPUTE MEANS
m0 = [];
m1 = [];
for j in range(0, cols, 1):
    m0.append(1);
    
for j in range(0, cols, 1):
    m1.append(1);
    
for i in range(0, rows, 1):
    if(trainlabels.get(i) != None and trainlabels[i] == 0):
        for j in range(0, cols, 1):
            m0[j] = m0[j] + data[i][j]
    if(trainlabels.get(i) != None and trainlabels[i] == 1):
        for j in range(0, cols, 1):
            m1[j] = m1[j] + data[i][j]

for j in range(0, cols, 1):
    m0[j] = m0[j]/n[0]
    m1[j] = m1[j]/n[1]
# COMPUTE STANDARD DEVIATION
s0 = [];
s1 = [];

for j in range(0, cols, 1):
    s0.append(0);

for j in range(0, cols, 1):
    s1.append(0);

for i in range(0, rows, 1):
    if(trainlabels.get(i) != None and trainlabels[i] == 0):
        for j in range(0, cols, 1):
            s0[j] = s0[j] + (data[i][j] - m0[j])**2;
    if(trainlabels.get(i) != None and trainlabels[i] == 1):
        for j in range(0, cols, 1):
            s1[j] = s1[j] + (data[i][j] - m1[j])**2;

for j in range(0, cols, 1):
    s0[j] = py.sqrt(s0[j]/(n[0] - 1))
    s1[j] = py.sqrt(s1[j]/(n[1] - 1))

# CLASSIFICATION
for i in range(0, rows, 1):
    if(trainlabels.get(i) == None):
        d0 = 0;
        d1 = 0;
        for j in range(0, cols, 1):
            d0 = d0 + ((data[i][j] - m0[j])/s0[j])**2;
            d1 = d1 + ((data[i][j] - m1[j])/s1[j])**2;
        if (d0 < d1):
            print("Naive Bayes Classifier: Label = 0", i);
        else:
            print("Naive Bayes Classifier: Label = 1", i);

        
        
        
