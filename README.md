# ATP

This is the official implementation of the following paper:

Wenxuan Bao, Tianxin Wei, Haohan Wang, Jingrui He. *Adaptive Test-Time Personalization for Federated Learning.*
NeurIPS 2023.

[\[Arxiv\]](https://arxiv.org/abs/2310.18816)
[\[Poster\]](https://github.com/baowenxuan/ATP/blob/master/material/ATP_poster.pdf)
[\[Slides\]](https://github.com/baowenxuan/ATP/blob/master/material/ATP_slides_short.pdf)

## Introduction

- We consider a novel setting named *Test-Time Personalized Federated Learning*, addressing the challenge of
  personalizing a global model to each *unparticipating client* during test-time, without requiring any labeled data.
- We propose *ATP*, which adaptively learns the adaptation rate for each module, enabling it to handle different types
  of distribution shifts among FL clients.

## Requirements

- python 3.8.5
- cudatoolkit 10.2.89
- cudnn 7.6.5
- pytorch 1.11.0
- torchvision 0.12.0
- numpy 1.18.5
- tqdm 4.65.0
- matplotlib 3.7.1

If you prefer generating the CIFAR-10C and CIFAR-100C by yourself, these packages may also be required:

- wandb 0.16.0
- scikit-image 0.17.2
- opencv-python 4.8.0.74

(This codebase should not be very sensitive to the version of packages.)

## Run

### CIFAR-10C Experiments

We consider three types of distribution shifts in our CIFAR-10C experiments: feature shift, label shift, and hybrid
shift.

```shell
cd ./exp/cifar10/${shift}
```

where `${shift}` should be replaced by `feat` (feature shift), `label` (label shift), or `hybrid` (hybrid shift). 

#### Generate Dataset

```shell
./data_prepare.sh
```

This shell script will partition the CIFAR-10 dataset to 300 clients (240 source clients and 60 clients), and save the
partition indices to `~/data/atp/partition/cifar10/`. When there are corruptions (feature shift and hybrid shift), we
also cache the corrupted dataset to `~/data/atp/cifar10` to save time. 

We also upload these 

#### Train Global Model with FedAvg

Before running ATP, we need to train a global model with source clients' training sets. We use FedAvg algorithm to train
the global model.

```shell
./pretrain_fedavg_${model}.sh
```

Here `${model}` specifies the model architecture we use. We used `resnet18` (ResNet-18) and `cnn` (shallow CNN) in our
paper.

#### Learn Adaptation Rates with ATP

```shell
./atp_train_${model}.sh
```

It also prints the evaluation result of ATP-batch in each iteration.

#### Test-Time Personalization with ATP-batch and ATP-online

```shell
./atp_test_${model}.sh
```

The results of

#### Expected Accuracies

Notice that this is the result with one seed, while we showed the results from five difference random seeds in our
paper. 

**ResNet-18**

| Algorithm     | Feature shift | Label shift | Hybrid shift |
|---------------|:-------------:|:-----------:|:------------:|
| No adaptation |     69.62     |    72.58    |    63.55     |
| ATP-batch     |     73.48     |    80.09    |    72.85     |
| ATP-online    |     73.83     |    81.72    |    75.34     |

**Shallow CNN**

| Algorithm     | Feature shift | Label shift | Hybrid shift |
|---------------|:-------------:|:-----------:|:------------:|
| No adaptation |     64.36     |    69.15    |    61.87     |
| ATP-batch     |     67.02     |    76.14    |    68.48     |
| ATP-online    |     67.22     |    78.38    |    70.86     |

### CIFAR-100C Experiments

Coming soon.

### Digits-5

Coming soon.

### PACS

Coming soon.

## Citation

If you are also interested in test-time personalization, please consider giving a star ⭐️ to our repo and citing our
paper:

```text
@inproceedings{
  bao2023adaptive,
  title={Adaptive Test-Time Personalization for Federated Learning},
  author={Wenxuan Bao and Tianxin Wei and Haohan Wang and Jingrui He},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=rbw9xCU6Ci}
}
```

