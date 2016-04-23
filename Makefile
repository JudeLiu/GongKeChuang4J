CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
SHVER = 3
#LIBS = -lblas

#all: train predict
all: train

lib: linear.o tron.o blas/blas.a
	SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; 
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train: train.cpp tron.o linear.o blas/blas.a
	$(CXX) $(CFLAGS) -std=c++11 -o train train.cpp tron.o linear.o $(LIBS)

predict: predict.o
	$(CXX) $(CFLAGS) -o predict predict.o tron.o linear.o $(LIBS)

predict.o: tron.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict.o -c predict.c tron.o linear.o $(LIBS)
	#$(CXX) $(CFLAGS) -o predict.o predict.c

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

random_train: random_train.o
	$(CXX) -std=c++11 $(CFLAGS) -o random_train random_train.cpp linear.o tron.o $(LIBS)

random_train.o: random_train.cpp  linear.o tron.o blas/blas.a
	$(CXX) -std=c++11 $(CFLAGS) -o random_train.o -c random_train.cpp

random_train_omp: random_train_omp.cpp linear.o tron.o blas/blas.a
	$(CXX) $(CFLAGS) -std=c++11 -o random_train_omp random_train_omp.cpp linear.o tron.o $(LIBS) -fopenmp

priori_train.o: priori_train.cpp 
	$(CXX) $(CFLAGS) -std=c++11 -o priori_train.o -c priori_train.cpp 

priori_train: priori_train.o linear.o tron.o blas/blas.a
	$(CXX) $(CFLAGS) -std=c++11 -o priori_train priori_train.o linear.o tron.o $(LIBS)

naive_train: naive_train.cpp linear.o tron.o blas/blas.a
	$(CXX) $(CFLAGS) -std=c++11 -o naive_train naive_train.cpp linear.o tron.o $(LIBS)

#liblinear_train_predict.o: liblinear_train_predict.cpp
	#$(CXX) $(CFLAGS) -c liblinear_train_predict.cpp -o liblinear_train_predict.o 

#main: main.cpp tron.o linear.o blas/blas.a
#main: main.o tron.o linear.o train.o blas/blas.a 
	#$(CXX) $(CFLAGS) main.cpp tron.o linear.o -o main $(LIBS)
	#$(CXX) $(CFLAGS) -o main main.o tron.o linear.o train.o $(LIBS)

#main.o: main.cpp train.o
	#$(CXX) $(CFLAGS) -c main.cpp -o main.o

#main.o: main.cpp tron.o linear.o blas/blas.a
	#$(CXX) $(CFLAGS) -c main.cpp -o main.o 

clean:
	make -C blas clean
	#make -C matlab clean
	rm -f *~ 
	rm -f *.o
	rm -f train 
	rm -f predict 
	rm -f random_train
	rm -f pred_out
	rm -f random_train_omp
	rm -f priori_train
	rm -f liblinear.so.$(SHVER)
	rm -f model/*
