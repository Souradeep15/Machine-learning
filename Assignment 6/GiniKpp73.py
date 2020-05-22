# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:17:06 2019

@author: Kishan Patel
"""
import sys
#import random as ran
#import math
#fifrom operator import mul

# Read The Data
datafile = sys.argv[1];
labelfile = sys.argv[2];
f = open(datafile);
l = f.readline();
data = [];

while (l != ''):
    a = l.split()
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

# Gini Index Implementation
gvals = [];
split = 0;
l3 = [0,0];
for j in range(0, cols, 1):
    gvals.append(l3);
ginit = 0;
col = 0;

for j in range(0, cols, 1):
    listcol = [item[j] for item in data]
    keys = sorted(range(len(listcol)), key = lambda k: listcol[k])
    listcol.sort();
    gv = [];
    prevgini = 0;
    prevrow = 0;
    for k in range(1, rows, 1):
        lsize = k;
        rsize = rows - k;
        lp = 0;
        rp = 0;

        for i in range(0, k, 1):
            if (trainlabels.get(keys[i]) == 0):
                lp += 1
        for m in range(k, rows, 1):
            if (trainlabels.get(keys[m]) == 0):
                rp += 1
        gini = (lsize/rows)*(lp/lsize)*(1 - lp/lsize)+(rsize/rows)*(rp/rsize)*(1-rp/rsize);
        gv.append(gini);
        prevgini = min(gv);

        if(gv[k - 1] == float(prevgini)):
            gvals[j][0] = gv[k - 1];
            gvals[j][1] = k;
    if(j == 0):
        ginit = gvals[j][0];
    if(gvals[j][0] <= ginit):
        ginit = gvals[j][0];
        col = j;
        split = gvals[j][1];
        if(split != 0):
            split = (listcol[split] + listcol[split - 1]) / 2;
print("gini = ", ginit, " col = ", col, " split = ", split);
