import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_test_split(X,Y,train_size,test_size):
    n,m = X.shape
    x = int(train_size*n)
    trainX = np.array([]).reshape(m,0)
    trainY = np.array([]).reshape(1,0)
    for i in range(x):
        trainX = np.c_[trainX,X[0]]
        X = np.delete(X,0,0)
        trainY = np.c_[trainY,Y[0]]
        Y = np.delete(Y,0,0)
    return trainX.T,X,trainY.T,Y


def train(dataset):

    np.random.shuffle(dataset)
    X, X_test, T, T_test = train_test_split(dataset[:,:-1], dataset[:,-1], train_size=0.70,test_size=0.30)
    #train

    n,m = X.shape
    W = np.random.rand(m+1)
    W = W.T
    X = np.append(np.ones((n,1)),X,axis=1)
    Y = [False for i in range(n)]
    T[T==0]=-1
    ita = 0.001

    counter=0
    while not all(Y) and counter < 1000000:
        i = counter%n
        pred = T[i]*(np.dot(W.T,X[i].T))
        if pred >= 0:
            Y[i] = True
        else:
            Y[i] = False
            err = T[i]*X[i].T
            W = W + ita*err
        counter+=1
    print("WEIGHTS: ", W)
    print("ITERATIONS: ", counter)
    print("TRAIN ACCURACY: ", sum(Y)/n)
    

    ## test

    n_test = X_test.shape[0]
    X_test = np.append(np.ones((n_test,1)),X_test,axis=1)
    T_test[T_test==0]=-1
    Y_test=T_test*(np.dot(W.T,X_test.T))
    O = Y_test>0
    print("TEST ACCURACY: ",sum(O)/n_test)
    print()


if __name__ == "__main__":
    
    datasets = {"dataset_LP_1.txt","dataset_LP_2.csv"}
    for i in datasets:
        print("DATASET :: ", i)
        dataset = np.loadtxt(i,delimiter=',')
        train(dataset)

