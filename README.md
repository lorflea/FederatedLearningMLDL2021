# Federated Learning MLDL-2021

This work is a comparison between the best known federated learning algorithms, the experiments are made CIFAR10 in a non-IID scenario.
Based on the following works:

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)


## Requirements
python>=3.6  
pytorch>=0.4

## Run

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --alg fedprox --model lenet --alpha 0.05 --epochs 50 --local_ep 10

`--all_clients` for averaging over all client models

To run SCAFFOLD enter the scaffold directory and follow the readme.


## Ackonwledgements
Scaffold Acknowledgements give to [Xtra-Computing](https://github.com/Xtra-Computing/NIID-Bench).


