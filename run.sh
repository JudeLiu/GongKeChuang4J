#./train -c 1 -s 2 data/myTrain.txt model
#./predict data/myTest.txt model pred_out
if [ $# -eq 1 ]; then 
    ./random_train_omp -q -c 1 -s 2 data/myTrain.txt data/myTest.txt
else
    ./random_train -q -c 1 -s 2 data/myTrain.txt data/myTest.txt
fi
#./predict data/myTest.txt model pred_out
