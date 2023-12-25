cd ../../../src || exit

gpu=3

dataset='cifar10'
num_clients=300
partition='step_2_16'
data_holdout=0.2
client_holdout=0.2
corruption="none"

model='cnn'
algorithm='atp'
gm_rounds=400
part_rate=0.25

lm_lr=0.1
lm_epochs=1
batch_size=20

partition_seed=0

for seed in {0..0}; do
  {
    load_model_path="../weights/cifar10/label/pretrain_fedavg_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
    history_path="../history/cifar10/label/atp_${model}_pseed_${partition_seed}_seed_${seed}.pkl"

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
      --gm_rounds ${gm_rounds} \
      --part_rate ${part_rate} \
      --lm_lr ${lm_lr} \
      --lm_epochs ${lm_epochs} \
      --batch_size ${batch_size} \
      --seed ${seed} \
      --cuda \
      --history_path ${history_path} \
      --load_model_path ${load_model_path}
  }
done