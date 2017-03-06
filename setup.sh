mkdir -p data
ln -s /ais/gobi4/mren/data/cifar-10 data/cifar-10
ln -s /ais/gobi4/mren/data/cifar-100 data/cifar-100
ln -s /ais/gobi4/mren/data/imagenet data/imagenet
mkdir -p logs/default
ln -s /u/$USER/public_html/results logs/public
mkdir -p /ais/gobi4/mren/$USER/results/resnet
ln -s /ais/gobi4/$USER/results/resnet results
