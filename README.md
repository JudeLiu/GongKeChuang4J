#Introduction
This project is the implementation of SJTU Gong Ke Chuang(工科创) 4J.

It implements a two-class classifier to classify which section a patent is in. (Actually it is a simplified version of the problem that Prof. Lu Baoliang did thorough research during his PhD. The original problem aims to classify a huge set of Japanese patents, which requires a multiclass classifier to classfy hierarchical labels, from top to bottom are section, class, subclass are group.) 

Since the training set is quite large(actually the training set provided by professor is not too large, say 11k, but the original training set is quite large indeed.), using SVM will cost too much time to train the neural network(NN), so we propose to use a open source tool LIBLINEAR(www.csie.ntu.edu/~cjlin/liblinear) to train the model.

#Requirement of the project
1. directly use LIBLINEAR to train and predict
2. train MIN-MAX model in one process and multi-process, randomly split subtasks
3. train MIN-MAX model in one process and multi-process, using priori knowledge to split subtasks
4. compute F1, and draw ROC
5. compare the time and performance
6. (optional) 
7. (optional)

#Components
- $\left.\begin{array}{l}
\mbox{linear.\*}\\\\
\mbox{tron.\*}\\\\
\mbox{blas/}\end{array}\right\\}$ copy from LIBLINEAR
- header.h : header file
- train.cpp : main function, train different models with different training methods according to the input parameter
- train_omp.cpp : parallel training
- Makefile : makefile on Linux
- model/ : an empty folder to contain saved model files.

#How to build
simply type
```
    cd /path/to/this/project
    make
```

#How to run
```
    cd /path/to/this/project
    mkdir data
    cp /path/to/the/training/set.txt data/
    cp /path/to/the/test/set.txt data/
    ./train n|r|p [options] train_set_name test_set_name
```
mkdir data and cp training set and test set are optional.

The first argument of ./train decides the training methods:
- n for naive LIBLINEAR training method
- r for random decomposition minmax method
- p for priori-knowledge-base decomposition minmax method

Simply type
```
    ./train
```
will print help message.

#Random decomposition MIN-MAX model
MIN-MAX model is proposed by Lu Baoliang, et al(see IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL.10, NO.5,SEPTEMBER 1999. Task Decomposition and Module Combination Based on Class Relations: A Modular Neural Network for Pattern Classification)

I implement random decomposition method to train min-max model. Using parameter "-s 5 -c 1 -n 2 -N 3" will achieve best performace. Actually from the result of experiment, letting subprob\_A be 1, subprob\_NA be 2 will achieve best performance, but in this case training set of label A is not decomposed, so I have to let subprob\_A to be 2 and subprob_NA to be 3, which is best among the rest cases.

#priori-knowledge-based decomposition MIN-MAX model
Using parameter "-s 2 -c 1" will achieve best performance.

