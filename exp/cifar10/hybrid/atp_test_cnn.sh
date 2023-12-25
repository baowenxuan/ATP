cd ../../../src || exit

gpu=3

dataset='cifar10'
num_clients=300
partition='step_2_16'
data_holdout=0.2
client_holdout=0.2
corruption="ood"

model='cnn'
algorithm='atptest'

batch_size=20

partition_seed=0

tests=('batch' 'online_avg')  # ATP-batch, ATP-online

for seed in {0..0}; do
  for i in {0,1}; do
    {
      echo ${tests[i]}
      load_model_path="../weights/cifar10/hybrid/pretrain_fedavg_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
      load_adapt_path="../history/cifar10/hybrid/atp_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
      load_adapt_idx=0
      load_adapt_round=-1

      history_path="../history/cifar10/hybrid/atp_test_${tests[i]}_pseed_${partition_seed}_seed_${seed}.pkl"

      CUDA_VISIBLE_DEVICES=${gpu} python main.py \
        --dataset ${dataset} \
        --num_clients ${num_clients} \
        --partition ${partition} \
        --data_holdout ${data_holdout} \
        --client_holdout ${client_holdout} \
        --partition_seed ${partition_seed} \
        --corruption ${corruption}\
        --model ${model} \
        --algorithm ${algorithm} \
        --test ${tests[i]} \
        --load_adapt_path ${load_adapt_path} \
        --load_adapt_idx ${load_adapt_idx} \
        --load_adapt_round ${load_adapt_round} \
        --batch_size ${batch_size} \
        --seed ${seed} \
        --cuda \
        --history_path ${history_path} \
        --load_model_path ${load_model_path}
    }
  done
done