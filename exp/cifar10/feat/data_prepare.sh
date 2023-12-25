cd ../../../src || exit

# CIFAR-10 feature shift experiments

dataset='cifar10'
num_clients=300
partition='stratified' # balanced label distribution for all clients
data_holdout=0.2
client_holdout=0.2
corruption="ood" # train and test use different set of distortions

partition_seed=0

python ./cifar_prepare.py \
  --dataset ${dataset} \
  --num_clients ${num_clients} \
  --partition ${partition} \
  --data_holdout ${data_holdout} \
  --client_holdout ${client_holdout} \
  --corruption ${corruption} \
  --partition_seed ${partition_seed}
