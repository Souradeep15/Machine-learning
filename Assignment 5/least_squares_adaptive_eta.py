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
# e = 0.001;
eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
stop = 0.001;
prevobj = 1000000000;
obj = prevobj - 10;
dp = 0;
best_eta = 1;

dellf = [];
for i in range(0, cols, 1):
    dellf.append(0);

while (prevobj - obj > stop):

    prevobj = obj;
    # Reset dellf to 0
    for j in range(0, cols, 1):
        dellf[j] = 0

    # Compute dellf
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            dp = sum(map(mul, w, data[i]));
            for j in range(0, cols, 1):
                dellf[j] += (trainlabels[i] - dp)*(data[i][j]);

    bestobj = 1000000000000;
    for k in range(0, len(eta_list), 1):
        e = eta_list[k]
        # Update w (locally)
        for j in range(0, cols, 1):
            w[j] = w[j] + e * dellf[j];
    
        # Compute Error
        error = 0;
        for i in range(0, rows, 1):
            if (trainlabels.get(i) != None):
                error += (trainlabels[i] - (sum(map(mul, w, data[i]))))**2;
        obj = error;

        # Set best obj/eta
        if(obj < bestobj):
            best_eta = e;
            bestobj = obj;

        # Remove w
        for j in range(0, cols, 1):
            w[j] = w[j] - e * dellf[j];
        #print(e, obj, prevobj);

    e = best_eta;
    # Update w
    for j in range(0, cols, 1):
            w[j] = w[j] + e * dellf[j];

    # Compute Error
    error = 0;
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            error += (trainlabels[i] - (sum(map(mul, w, data[i]))))**2;
    obj = error;
    #print("best_eta=", best_eta, " Objective=", error);
    
#print("w = ");
nw = 0;
for j in range(0, cols - 1, 1):
    nw += w[j]**2;
    #print(w[j]);
#print("\n");
nw = math.sqrt(nw);
dorg = abs(w[len(w) - 1]/nw);
#print("Distance to the Origin = ", dorg);

# Prediction
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        dp = (sum(map(mul, w, data[i])));
        if (dp > 0):
            print("1", i);
        else:
            print("0", i);
