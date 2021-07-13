#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms, models
import torch

from utils.sampling import cifar_iid, cifar_noniid, cifar_noniid_unequal, get_dict_users
from utils.options import args_parser
from models.Update import LocalUpdate, LocalProxUpdate
from models.Nets import LeNet
from models.Fed import FedAvg
from models.test import test_img 

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users, num_class_users = get_dict_users(args.alpha)
    
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'lenet':
        net_glob = LeNet().to(args.device)

    elif args.model == 'googlenet':
        net_glob = models.googlenet(pretrained=True).to(args.device)

   
    elif args.model == 'resnet18':
        net_glob = models.resnet18(pretrained=True).to(args.device)

    elif args.model == 'resnet50':
        net_glob = models.resnet50(pretrained=True).to(args.device)

    elif args.model == 'alexnet':
        net_glob = AlexNet().to(args.device)
 
    else:
        exit('Error: unrecognized model')


    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    accur = []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            len_locals = []
            num_class_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            if args.alg == 'fedavg':
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            elif args.alg == 'fedprox':
                local = LocalProxUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(server_net=copy.deepcopy(net_glob).to(args.device), net=copy.deepcopy(net_glob).to(args.device), num_class=num_class_users[idx])

            else:
                exit('Error: unrecognized aggregation algorithm')

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
                len_locals.append(len(dict_users[idx]))
                #num_class_locals.append(num_class_users[idx])

            loss_locals.append(copy.deepcopy(loss))

        w_glob = FedAvg(w_locals, len_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print('Round {:3d}, Testing accuracy {:.3f}'.format(iter, acc_test))
        accur.append(float(acc_test))
        print(accur)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
