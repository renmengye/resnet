DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NAME=${DIR##*/}
mkdir -p data
ln -s /ais/gobi4/mren/data/cifar-10 data/cifar-10
ln -s /ais/gobi4/mren/data/cifar-100 data/cifar-100
ln -s /ais/gobi4/mren/data/imagenet data/imagenet
mkdir -p logs/default
ln -s /u/$USER/public_html/results logs/public
mkdir -p /ais/gobi5/$USER/results/$NAME
ln -s /ais/gobi5/$USER/results/$NAME results
