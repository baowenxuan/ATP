cd ../../../src || exit

gpu=0

dataset='cifar10'
num_clients=300
partition='stratified'
data_holdout=0.2
client_holdout=0.2
corruption="ood"

model='resnet18'
algorithm='fedavg'
gm_rounds=200
part_rate=1.0

lm_lr=0.01
lm_epochs=1
batch_size=20

partition_seed=0

for seed in {0..0}; do
  {
    save_model_path="../weights/cifar10/feat/pretrain_${algorithm}_${model}_pseed_${partition_seed}_seed_${seed}.pkl"
    history_path="../history/cifar10/feat/pretrain_${algorithm}_${model}_pseed_${partition_seed}_seed_${seed}.pkl"

    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset ${dataset} \
      --num_clients ${num_clients} \
      --partition ${partition} \
      --data_holdout ${data_holdout} \
      --client_holdout ${client_holdout} \
      --partition_seed ${partition_seed} \
      --corruption ${corruption} \
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
      --save_model_path ${save_model_path}
  }
done
