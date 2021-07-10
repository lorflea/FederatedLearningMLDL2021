# Federated Learning MLDL-2021

This work is a comparison between the best known federated learning algorithms, the experiments are made CIFAR10 in a non-IID scenario.
Based on the following works:

[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
<br>
[Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)
<br>
[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)


## Requirements
python>=3.6  
pytorch>=0.4

## Run


See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --model lenet --alpha 0.05 --epochs 50 

`--all_clients` for averaging over all client models


## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## Ackonwledgements
Acknowledgements give to [shaoxiongji](https://github.com/shaoxiongji).


