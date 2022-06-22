#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from tst.encoder import Encoder
from tst.utils import generate_original_PE, generate_regular_PE

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class baseNetwork(nn.Module):
    """
    DeepCORAL network as defined in the paper.
    Network architecture based on following repository:
    https://github.com/SSARCandy/DeepCORAL/blob/master/models.py
    :param num_classes: int --> office dataset has 31 different classes
    """
    def __init__(self,  d_input, 
                      d_model, 
                      q, v, h, N, 
                      attention_size, 
                      chunk_mode,
                      pe, 
                      pe_period=45,
                      num_classes=5,
                      bottleneck_dim=256):
        super(baseNetwork, self).__init__()
        self.sharedNetwork = AlexNet(d_input, d_model, q, v, h, N, attention_size=attention_size, chunk_mode=chunk_mode, pe=pe, pe_period=45)
        self.bottleneck = nn.Linear(128,bottleneck_dim)
        self.fc8 = nn.Linear(bottleneck_dim, num_classes) # fc8 activation

        # initiliaze fc8 weights according to the CORAL paper (N(0, 0.005))
        self.fc8.weight.data.normal_(0.0, 0.005)

    def forward(self, source): # computes activations for BOTH domains
        features = self.sharedNetwork(source)
        features = self.bottleneck(features)
        outputs = self.fc8(features)
        return features, outputs



class AlexNet(nn.Module):
    """
    AlexNet model obtained from official Pytorch repository:
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    """
    def __init__(self, 
                 d_input: int,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = 45):
        super(AlexNet, self).__init__()
        self._d_model = d_model
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                          q,
                          v,
                          h,
                          attention_size=attention_size,
                          dropout=dropout,
                          chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None

        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = self._d_model * 45, out_features = 128),
                nn.Dropout(),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(in_features = 128, out_features = 128),
                nn.ReLU(inplace=True), # take fc8 (without activation)
            )

    def forward(self, input_data):
        K = input_data.shape[1]

        # Embedding module
        encoding = self._embedding(input_data)

        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)
        
        # Encoding
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            
        feature = torch.tanh(encoding)
        
        # Classification on source domain data
        x = self.classifier(feature)
        return x

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork obtained from official CDAN repository:
    https://github.com/thuml/CDAN/blob/master/pytorch/network.py
    """
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()

        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.batchNM1 = nn.BatchNorm1d(hidden_size)
        self.batchNM2 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        #print("inside ad net forward",self.training)
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.batchNM1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.batchNM2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y


    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
