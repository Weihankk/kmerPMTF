#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:57:18 2023

@author: weihan
"""

import time
import torch
import argparse
import pandas as pd
from model.DeepNeuralNetworks import DeepNeuralNetworks

def set_args():
    parser = argparse.ArgumentParser(description='Graph construct by genome and miRNA information')
    
    parser.add_argument('--prefix', type=str,
                        default='Prunus_persica',
                        help='The output file prefix, MUST same as step1')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='The device, cuda/cpu')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int,
                        default=500,
                        help='Number of training epochs')
    parser.add_argument('--D', type=int,
                        default=256,
                        help='Recommended to try 16/32/64/128/256/512/1024, higher D corresponds to slower speeds and exponential memory consumption')
    
    args = parser.parse_args()
    
    return args

def localtime():
    return(time.strftime(' %Y-%m-%d %H:%M:%S\n --> ', time.localtime()))

if __name__ == '__main__':
    args = set_args()
    
    args_dict = vars(args)
    args_command = 'python step2_encoded.py'
    for i in args_dict.keys():
        args_command = args_command + ' --' + i
        args_command = args_command + ' ' + str(args_dict[i])
    
    print('Command: {}'.format(args_command))
    
    # 1. Import data
    print('\033[1;36m{}\033[0m'.format(localtime()),'Import node attr data ...\n')
    node_attr = pd.read_csv('./process_data/{}_node_attr.csv'.format(args.prefix), header=0, index_col=0)
    node_attr_train = torch.tensor(node_attr.values, dtype=torch.float32).to(args.device)
    
    # 2. DNN 
    print('\033[1;36m{}\033[0m'.format(localtime()),'Start training ...\n')
    model = DeepNeuralNetworks(node_attr_train.shape[1], args.D).to(args.device)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        
        encoded, decoded = model(node_attr_train)
        loss = criterion(decoded, node_attr_train)
        loss.backward()
        
        optimizer.step()
        
        if (epoch+1)%50 == 0:
            print('\033[5;33m     Deep Neural Networks: Epoch:[{}/{}], Loss:{:.5f}\033[0m'.format(epoch+1, args.epoch, loss.item()))
        
    with torch.no_grad():
        node_attr_encoded, node_attr_decoded = model(node_attr_train)
        
        node_attr_encoded = node_attr_encoded.detach().cpu().numpy()
        
        node_attr_output = pd.DataFrame(node_attr_encoded, index=node_attr.index.tolist())
        
        node_attr_output.to_csv('./process_data/{}_D{}_node_attr_encode.csv'.format(args.prefix, args.D), header=True, index=True, float_format='%.10f')
        
        
