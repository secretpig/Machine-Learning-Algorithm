#!/usr/bin/python3

import sys
import numpy as np

def readdata(filename):
    y=[]
    x=[]
    file=open(filename,'r')
    lines=file.readlines()
    for line in lines:
        data=line.split()
        y.append(int(data[0]))
        index=[]
        for d in data[1:]:
            index.append(int(d[:-2]))
        temp=[0 if i not in index else 1 for i in range(123)]
        temp.append(1)
        x.append(temp)
    file.close
    return x,y

def perceptron(w,x,y):         
    Len=len(y)
    for n in range(Len):
        if np.dot(w,x[n])>0:
            t=1
        elif np.dot(w,x[n])<0:
            t=-1
        else:
            t=0
        if y[n]!=t:
            for i in range(124):
                w[i]=w[i]+y[n]*x[n][i]
    return w

def accuracy(x,y,w):
    count=0
    for n in range(len(y)):
        if np.dot(w,x[n])>0:
            t=1
        elif np.dot(w,x[n])<0:
            t=-1
        else:
            t=0
        if t==y[n]:
            count+=1
    return count/len(y)

#read commondline
iterate=int(sys.argv[-1])
if '--nodev' not in sys.argv:
    best=True
else:
    best=False

# train prepare
#iterate=30
traindata='/u/cs246/data/adult/a7a.train'
x1,y1=readdata(traindata)

devdata='/u/cs246/data/adult/a7a.dev'
x3,y3=readdata(devdata)
alpha=0.0
iterations=0
#
w=[0 for x in range(124)]
for i in range(iterate):
    w=perceptron(w,x1,y1)
    temp=accuracy(x3,y3,w)
    if alpha<temp:
        alpha=temp
        wbest=w[:]
        iterations=i+1

#test
testdata='/u/cs246/data/adult/a7a.test'
x2,y2=readdata(traindata)
if best==False:
    rate=accuracy(x2,y2,w)
    solution=w
else:
    rate=accuracy(x2,y2,wbest)
    solution=wbest
    print('Best iterations: %d' % iterations)

#print
print('Test accuracy: %f' % rate)
print('Feature weights (bias last): ',end='')
for i in solution:
    print('%.1f' % i,end=' ')
print('')