cd ../../../src || exit

# CIFAR-10 label shift experiments

dataset='cifar10'
num_clients=300
partition='step_2_16' # 2 major class, major : minor = 80 : 5
data_holdout=0.2
client_holdout=0.2
corruption="none"

partition_seed=0

python ./cifar_prepare.py \
  --dataset ${dataset} \
  --num_clients ${num_clients} \
  --partition ${partition} \
  --data_holdout ${data_holdout} \
  --client_holdout ${client_holdout} \
  --corruption ${corruption} \
  --partition_seed ${partition_seed}
