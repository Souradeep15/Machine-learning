# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:42:27 2019

@author: Kishan Patel
"""
import sys;
import math;
import random as rnd;
from collections import defaultdict

def dist(x, K):
    s = sum([(m-n)**2 for m,n in zip(x,K)])
    s_sqrt = math.sqrt(s);
    return s

def fclst(x, K):
    d = defaultdict(list);
    m = [];
    for i in range(0, len(x)):
        t = [];
        for j in range(0, len(K)):
            t.append(dist(x[i],K[j]));
        indi = t.index(min(t));
        d[indi].append(x[i])
    for k in d:
        t = d[k]
        m.append([sum(n)/len(n) for n in zip(*t)])
    return m

def hs(O,N):
    return(set([tuple(a) for a in O]) == set([tuple(a) for a in N]))

def km(X,K):
    m = rnd.sample(X,K);
    while True:
        mn = fclst(X, m);
        if hs(m,mn):
            m = mn
            break
        m = mn;
    print('\nLabels and Indicies');
    for i in range(0, len(X)):
        t = [];
        for j in range(0, len(m)):
            t.append(dist(X[i],m[j]))
        indi = t.index(min(t));
        print('{} {}'.format(indi, i));
    return m


datafile = sys.argv[1];
K = int(sys.argv[2]);
f = open(datafile, 'r');
data = [];
l = f.readline();
#Read the Data File
while(l != ''):
    a = l.split();
    l2 = [];
    for j in range(0, len(a), 1):
        l2.append(float(a[j]));
    data.append(l2);
    l = f.readline();
rows = len(data);
cols = len(data[0]);
f.close();
m = km(data, K);

for i in range(0, K, 1):
    print('Cluster {} = {}'.format(i,m[i]));


