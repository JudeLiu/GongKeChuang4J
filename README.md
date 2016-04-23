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

#Random decomposition MIN-MAX model
MIN-MAX model is proposed by Lu Baoliang, et al(see IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL.10, NO.5,SEPTEMBER 1999. Task Decomposition and Module Combination Based on Class Relations: A Modular Neural Network for Pattern Classification)

I implement random decomposition method to train min-max model. Using parameter "-s 5 -c 1 -n -N" will achieve best performace. Actually from the result of experiment, letting subprob_A be 1, subprob_NA be 2 will achieve best performance, but in this case training set of label A is not decomposed, so I have to let subprob_A to be 2 and subprob_NA to be 3, which is best among the rest cases.

C

#priori-knowledge-based decomposition MIN-MAX model
Using parameter "-s 2 -c 1" will achieve best performance.

