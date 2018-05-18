#!usr/bin/python
# -*- coding:UTF-8 -*-
#Created on: 2018/3/28
#author: Xiuze Zhou
#e-mail: zhouxiuze@foxmail.com
 
#-------------------------FUNCTION---------------------------#
from pylab import *
import numpy as np
import random
import math


def g(x):
    res=1/(1+math.exp(-x))
    return res


def dg(x):
    res=-math.exp(x)/(1+math.exp(x))**2
    return res


def data2mat(data, N, M):
    # get the indicator function
    I= [ [ 0 for i in range(M) ] for j in range(N) ]
    for d in data:
        u=d[0]
        i=d[1]
        r=d[2]
        I[u][i]=1
    I=np.array(I)   
    return I


def SGD(train,test,N,M,Level,eta,K,lambda_1,lambda_2,lambda_3,Step):
    # train: train data
    # test: test data
    # N:the number of user
    # M:the number of item
    # Level:the max value of rating
    # eta: the learning rata
    # K: the number of latent factor
    # lambda_1,lambda_2,lambda_3: regularization parameters
    # Step: the max iteration
    Y = np.random.normal(0, 1, (N, K))
    V = np.random.normal(0, 1, (M, K))
    W = np.random.normal(0, 1, (M, K))
    U = Y
    I = data2mat(train, N, M) # get the indicator function
    rmse=[]
    rms=RMSE(U,V,test,Level)
    rmse.append(rms)
    for ste in range(Step):
        for data in train:
            u=data[0]
            i=data[1]
            r=data[2]
            rui=(r-1)/(Level-1);

            Iu=I[u]
            Yu=Y[u]
            Vi=V[i]

            U[u]=Yu+np.dot(Iu,W)/sum(Iu)
            Uu=U[u]
            
            gui=g(np.dot(Uu,Vi.T))
            dgui=dg(np.dot(Uu,Vi.T))
            eui=rui- gui
            
            Y[u]=(1-eta*lambda_1)*Yu+eta*eui*dgui*Vi
            V[i]=(1-eta*lambda_2)*Vi+eta*eui*dgui*Uu
            W=(1-eta*lambda_3)*W+eta*eui/sum(Iu)*np.tile(Vi,(M,1))

        rms=RMSE(U,V,test,Level)
        rmse.append(rms)
        
        print ste
    return rmse,U,V

           
def RMSE(U,V,test,Level):
    count=len(test)
    sum_rmse=0.0
    for t in test:
        u=t[0]
        i=t[1]
        r=t[2]
        pr= g(np.dot(U[u],V[i].T))*(Level-1)+1
        sum_rmse+=np.square(r-pr)
    rmse=np.sqrt(sum_rmse/count)
    return rmse


def Load_data(filedir,ratio):
    user_set={}
    item_set={}
    rating_set={}
    N=0;#the number of user
    M=0;#the number of item
    Level=0;# the max value of rating
    u_idx=0
    i_idx=0
    r_idx=0
    data=[]
    f = open(filedir)
    for line in f.readlines():
        u,i,r,t=line.split()
        if int(u) not in user_set:
            user_set[int(u)]=u_idx
            u_idx+=1
        if int(i) not in item_set:
            item_set[int(i)]=i_idx
            i_idx+=1
        if int(r) not in rating_set:
            rating_set[int(r)]=r_idx
            r_idx+=1
        data.append([user_set[int(u)],item_set[int(i)],int(r)])
    f.close()
    N=u_idx
    M=i_idx
    Level=r_idx

    np.random.shuffle(data)
    train=data[0:int(len(data)*ratio)]
    test=data[int(len(data)*ratio):]
    return Level,N,M,train,test


def Figure(rmse):
    fig=plt.figure('RMSE')
    x = range(len(rmse))
    plot(x, rmse, color='r',linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    show()
    
#----------------------------SELF TEST----------------------------#
 
def main():
    dir_data="./u.data"
    ratio=0.8
    Level,N,M,train,test=Load_data(dir_data,ratio)
        
    eta=0.005
    K=10
    lambda_1=1
    lambda_2=1
    lambda_3=1
    Step=30
    rmse,U,V=SGD(train,test,N,M,Level,eta,K,lambda_1,lambda_2,lambda_3,Step)
    print rmse[-1];
    Figure(rmse)
    
         
if __name__ == '__main__': 
    main()
