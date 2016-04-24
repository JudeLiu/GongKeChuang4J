CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
SHVER = 3
#LIBS = -lblas

#all: train predict
all: train train_omp

lib: linear.o tron.o blas/blas.a
	SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; 
	$(CXX) $${SHARED_LIB_FLAG} linear.o tron.o blas/blas.a -o liblinear.so.$(SHVER)

train_omp: train_omp.cpp
	$(CXX) $(CFLAGS) -fopenmp -std=c++0x -o train_omp train_omp.cpp tron.cpp linear.cpp blas/*.c

train: train.cpp
	$(CXX) $(CFLAGS) -std=c++0x -o train train.cpp tron.cpp linear.cpp blas/*.c

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

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
