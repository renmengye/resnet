RESULTS_STORAGE=/ais/gobi4/$USER/results
DATA_STORAGE=/ais/gobi4/mren/data
#DATA_STORAGE=/home/mren/data
ln -s $DATA_STORAGE/cifar-10 data/cifar-10
ln -s $DATA_STORAGE/cifar-100 data/cifar-100
mkdir -p logs/default
mkdir -p results
mkdir -p $RESULTS_STORAGE
ln -s $RESULTS_STORAGE results
