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
        temp=[0 if i+1 not in index else 1 for i in range(123)]
        x.append(temp)
    file.close
    return x,y

def svm(C,x,y,t):
    w=[0 for x in range(123)]
    learning=0.1
    b=0    
    length=len(y)
    M=length
    for j in range(t):
        for n in range(M):
            if (np.dot(w,x[n])+b)*y[n]>=1:
                for i in range(123):
                    w[i]=w[i]-learning*w[i]/M
            else:
                for i in range(123):
                    w[i]=w[i]-learning*(w[i]/M-C*y[n]*x[n][i])
                b=b+learning*C*y[n]
    return w,b  

def accuracy(x,y,w,b):
    count=0
    for n in range(len(y)):
        if (np.dot(w,x[n])+b)*y[n]>0:
            count+=1
    return count/len(y)

traindata='/u/cs246/data/adult/a7a.train'
x1,y1=readdata(traindata)

testdata='/u/cs246/data/adult/a7a.test'
x2,y2=readdata(testdata)

devdata='/u/cs246/data/adult/a7a.dev'
x3,y3=readdata(devdata)

t=int(sys.argv[-3])
c=float(sys.argv[-1])
w,b=svm(c,x1,y1,t)
r1=accuracy(x1,y1,w,b)
r2=accuracy(x2,y2,w,b)
r3=accuracy(x3,y3,w,b)

result=[b]+w
print('EPOCHS: %d' % t)
print('CAPACITY: %f' % c)
print('TRAINING_ACCURACY: %f' % r1)
print('TEST_ACCURACY: %f' % r2)
print('DEV_ACCURACY: %f' % r3)
print('FINAL_SVM: ',end='')
print(result)
