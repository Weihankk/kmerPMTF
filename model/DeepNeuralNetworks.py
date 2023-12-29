#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:48:43 2023

@author: weihan
"""

from torch import nn

class DeepNeuralNetworks(nn.Module):
    def __init__(self, col, D):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(col, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1536),
            nn.Tanh(),
            nn.Linear(1536, 1024),
            nn.Tanh(),
            nn.Linear(1024, D),
            nn.Tanh()
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(D, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1536),
            nn.Tanh(),
            nn.Linear(1536, 2048),
            nn.Tanh(),
            nn.Linear(2048, col),
            nn.Tanh(),
            )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
