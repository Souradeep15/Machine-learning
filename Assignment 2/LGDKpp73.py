# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:42:12 2019

@author: Kishan Patel
"""

import math
import sys
import random as ran
from operator import mul

# Get The Data and Labels
datafile = sys.argv[1];
labelfile = sys.argv[2];
f = open(datafile);
l = f.readline();
i = 0;
data = [];

while(l != ''):
    a = l.split();
    l2 = [];
    for j in range(0, len(a), 1):
        l2.append(float(a[j]));
    data.append(l2);
    l = f.readline();
    data[i].append(1);
    i += 1;

rows = len(data);
cols = len(data[0]);

f = open(labelfile);
trainlabels = {};
n = [];
n.append(0);
n.append(0);
l = f.readline();
while (l != ''):
    a = l.split();
    trainlabels[int(a[1])] = int(a[0]);
    l = f.readline();
    n[int(a[0])] += 1;
    if (trainlabels[int(a[1])] == 0):
        trainlabels[int(a[1])] = -1;

# Initilization

w = [];

for j in range(0, cols, 1):
    w.append(((0.02 * ran.random()) - 0.01));
    
# Gradient Descent Iteration
e = 0.001;
stop = 0.001;
erold = 0;
dp = 0;

while True:
    # Compute dellf
    dellf = [];
    for i in range(0, cols, 1):
        dellf.append(0);
    
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            dp = sum(map(mul, w, data[i]));
            for j in range(0, cols, 1):
                dellf[j] += (trainlabels[i] - dp)*(data[i][j]);
    # Update w
    for j in range(0, cols, 1):
        w[j] = w[j] + e * dellf[j];
    
    # Compute Error
    error = 0;
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            error += (trainlabels[i] - (sum(map(mul, w, data[i]))))**2;
    # print("\nerror = ", error);
    if(abs(erold - error) <= stop):
        break;
    erold = error;
    
print("w = ");
nw = 0;
for j in range(0, cols - 1, 1):
    nw += w[j]**2;
    print(w[j]);
print("\n");
nw = math.sqrt(nw);
dorg = abs(w[len(w) - 1]/nw);
print("Distance to the Origin = ", dorg);

# Prediction
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        dp = (sum(map(mul, w, data[i])));
        if (dp > 0):
            print("1", i);
        else:
            print("0", i);
