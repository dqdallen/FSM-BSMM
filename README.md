# Geometric Imbalanced Deep Learning with Feature Scaling and Boundary Samples Mining
Zhe Wang, Qida Dong, Wei Guo, Dongdong Li, Jing Zhang, Wenli Du
-------------------------------------------------------------
This is the official implementation of FSM-BSMM in the paper Geometric Imbalanced Deep Learning with Feature Scaling and Boundary Samples Mining in PyTorch.
Our codes is rewritten on the basis of [LDAM-DRW](https://github.com/kaidic/LDAM-DRW)
## Dataset
Imbalanced CIFAR. The original data will be downloaded and converted by imbalancec_cifar.py.
## train
To train the CE on long-tailed imbalance with ratio of 100
```
!python /content/drive/MyDrive/mm/cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --epochs 200
```
To train the FSM-BSMM on long-tailed imbalance with ratio of 100
```
!python /content/drive/MyDrive/mm/cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type MyLoss --epochs 200
```
