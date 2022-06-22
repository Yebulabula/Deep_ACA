#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from loss import CDAN, Entropy
from model import calc_coeff
import sys

def train(model, ad_net, source_loader, target_loader,
          optimizer, epoch, lambda_factor, cuda=False):
    """
    This method fits the network params one epoch at a time.
    Implementation based on:
    https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    """
    results = [] # list to append loss values at each epoch

    # first cast into an iterable list the data loaders
    # data_source: (batch_size, len_seq, channels)
    # data_target: (batch_size)

    # memory leakage
    # print("memory leak")
    source = list(enumerate(source_loader))
    # print(source)
    target = list(enumerate(target_loader))
    train_steps = min(len(source), len(target))

    # start batch training
    for batch_idx in range(train_steps):
        # fetch data in batches
        # _, source_data -> torch.Size([128, 45, 1]), labels -> torch.Size([128])
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx] # unsupervised learning

        if source_data.shape[0] != target_data.shape[0]:
            break

        if cuda:
            # move to device
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        # create pytorch variables, the variables and functions build a dynamic graph of computation
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        # reset to zero optimizer gradients
        optimizer.zero_grad()

        # do a forward pass through network (recall DeepCORAL outputs source, target activation maps)
        src_features, src_ouputs = model(source_data)
        tgt_features, tgt_ouputs = model(target_data)
        features = torch.cat((src_features, tgt_features), dim=0)
        outputs = torch.cat((src_ouputs, tgt_ouputs), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        entropy = Entropy(softmax_out)

        # compute losses (classification and coral loss)
        classification_loss = torch.nn.functional.cross_entropy(src_ouputs, source_label)
        transfer_loss = CDAN([features, softmax_out], ad_net)

        # compute total loss (equation 6 paper)
        total_loss = classification_loss + lambda_factor * transfer_loss

        # compute gradients of network (backprop in pytorch)
        total_loss.backward()

        # update weights of network
        optimizer.step()

        # append results for each batch iteration as dictionaries
        results.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': lambda_factor,
            'cdan_loss': transfer_loss.item(), # coral_loss.data[0],

            'classification_loss': classification_loss.item(),  # classification_loss.data[0],
            'total_loss': total_loss.item() # total_loss.data[0]
        })

        # print training info
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], Lambda value: %f, Classification loss: %f, CDAN loss: %f, Total_Loss: %f' \
              % (epoch, batch_idx + 1, train_steps, lambda_factor,
                 classification_loss.item(), transfer_loss.item(), total_loss.item()))
        sys.stdout.flush()
    return results