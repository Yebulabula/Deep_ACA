#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TODO: get curves
# TODO: try different combinations among datasets

from __future__ import division
import warnings

from train import train
from test1 import test
import torch
from utils import  _split_data_labels, _str_to_int, df_to_tensor, save_log, arr_to_tensor, plot_learning_curves, plot_cm
from model import  AdversarialNetwork, baseNetwork
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from data_loader import dataset
import numpy as np
from imblearn.over_sampling import SMOTE 
from torch.autograd import Variable
import pandas as pd


# set model hyperparameters (paper page 5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = False

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
d_model = 16 #32  Lattent dim
q = 4  # Query size
v = 4  # Value size
h = 2 #4   Number of heads
N = 1  # Number of encoder and decoder to stack
attention_size = 15  # Attention window size
pe = "regular"  # Positional encoding
chunk_mode = None
d_input = 1 # the number of channel of each data.
epochs = 1
batch_size_source = 128
batch_size_target = 128
num_classes = 7

def main():
    """
    This method puts all the modules together to train a neural network
    classifier using CORAL loss.

    Reference: https://arxiv.org/abs/1607.01719
    """
    warnings.filterwarnings("ignore")

    with open('/Users/yemao/Downloads/CDAN/dataset/plex7/new_plex7_df_ac.csv', mode='r') as gblock:
        data = pd.read_csv(gblock)
        
    data = data[data['PrimerMix'] == "PM7.2151"]

    gblock_data = data[data["Conc"] != 'unknown']
    source_data = gblock_data[gblock_data["Exp_ID"].isin(['20210707_01', '20210707_03'])]
    target_data = gblock_data[gblock_data["Exp_ID"] == '20210721_02']
   

    source_X_df, source_Y_df = _split_data_labels(source_data)
    target_X_df, target_Y_df = _split_data_labels(target_data)

    # sm = SMOTE(random_state=42)
    # source_X_df, source_Y_df = sm.fit_resample(source_X_df, source_Y_df)
    
    source_X, source_Y = df_to_tensor(source_X_df),  torch.from_numpy(_str_to_int(source_Y_df))
    target_X, target_Y = df_to_tensor(target_X_df),  torch.from_numpy(_str_to_int(target_Y_df))

    # create dataloaders (Experiment 1 & 2 as source, Experiment 3 as target)
    source_dataset = dataset(source_X.to(device), source_Y.to(device))
    target_dataset = dataset(target_X.to(device), target_Y.to(device))

    source_loader = DataLoader(dataset = source_dataset, batch_size = batch_size_source, shuffle = True)
    target_loader = DataLoader(dataset = target_dataset, batch_size = batch_size_target, shuffle = True)


    bottleneck_dim = 256
    # define generator and classifier network
    model = baseNetwork(d_input, 
                        d_model, 
                        q, v, h, N, 
                        attention_size, 
                        chunk_mode,
                        pe, 
                        pe_period=45,
                        num_classes= num_classes,
                        bottleneck_dim = bottleneck_dim)

    # define discriminator network
    ad_net = AdversarialNetwork(bottleneck_dim * num_classes, 256)

    model.train(True)
    ad_net.train(True)

    # define optimizer pytorch: https://pytorch.org/docs/stable/optim.html
    # specify learning rates per layers:
    # 10*learning_rate for last two fc layers according to paper
    optimizer = torch.optim.Adam([
        {"params": model.sharedNetwork.parameters()},
        {"params": model.fc8.parameters(), "lr": LEARNING_RATE},
        {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
    ], lr=LEARNING_RATE)

    # move to CUDA if available
    if CUDA:
        model = model.to(device)
        ad_net = ad_net.to(device)
        print("using cuda...")

    print("model type:", type(model))

    # store statistics of train/test
    training_s_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    # start training over epochs
    print("running training for {} epochs...".format(epochs))

    best_acc_s = 0
    best_acc_t = 0
    for epoch in range(0, epochs):
        # compute lambda value from paper (eq 6)
        lambda_factor = (epoch+1)/epochs 
        # lambda_factor = 0

        # run batch trainig at each epoch (returns dictionary with epoch result)
        result_train = train(model, ad_net, source_loader, target_loader,
                             optimizer, epoch+1, lambda_factor, CUDA)

        # print log values
        print()
        print("[EPOCH] {}: Classification: {:.6f}, CDAN loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch+1,
                sum(row['classification_loss'] / row['total_steps'] for row in result_train),
                sum(row['cdan_loss'] / row['total_steps'] for row in result_train),
                sum(row['total_loss'] / row['total_steps'] for row in result_train),
            ))

        training_s_statistic.append(result_train)

        # perform testing simultaneously: classification accuracy on both dataset
        test_source = test(model, source_loader, epoch, CUDA)
        test_target = test(model, target_loader, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)
   
        print("[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_source['average_loss'],
                test_source['correct_class'],
                test_source['total_elems'],
                test_source['accuracy %'],
            ))

        print("[Test Target]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch+1,
                test_target['average_loss'],
                test_target['correct_class'],
                test_target['total_elems'],
                test_target['accuracy %'],
        ))

        if test_target['accuracy %'] > best_acc_t:
            best_acc_s = test_source['accuracy %']
            best_acc_t = test_target['accuracy %']
            torch.save(model, '{0}/CDAN_ACA_Exp_DA(1).pth'.format('/Users/yemao/Downloads/CDAN/CDAN_ACA/model'))  



    Activities = ['ADE', 'C22', 'CHK', 'CNL', 'COC', 'COV', 'MER']

    model = torch.load('{0}/CDAN_ACA_Exp_DA(1).pth'.format('/Users/yemao/Downloads/CDAN/CDAN_ACA/model'))
    model.eval()

    true = pd.DataFrame(index= target_X_df.index, columns= ['col1'])
    preds = pd.DataFrame(index= target_X_df.index, columns= ['col1'])

    pred_list = torch.tensor([]).to(device)
    label_list = torch.tensor([]).to(device)

    for data, label in target_loader:
        data, label = data.to(device), label.to(device)
        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
        data, label = Variable(data, volatile=True), Variable(label)
        _, output = model(data) # just use one ouput of DeepCORAL

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        
        pred = torch.flatten(pred)
        label = torch.flatten(label)
        
        pred_list = torch.cat((pred_list, pred), 0).to(device)
        label_list = torch.cat((label_list, label), 0).to(device)

    true['col1'] = label_list.cpu().numpy()
    preds['col1'] = pred_list.cpu().numpy()

    plot_cm(label_list.cpu().numpy(), pred_list.cpu().numpy(), Activities)
    print(classification_report(label_list.cpu().numpy(), pred_list.cpu().numpy(), digits = 5, target_names = Activities))

    # save results
    print("saving results...")
    save_log(training_s_statistic, '/Users/yemao/Downloads/CDAN/CDAN_ACA/training_s_statistic.pkl')
    save_log(testing_s_statistic, '/Users/yemao/Downloads/CDAN/CDAN_ACA/testing_s_statistic.pkl')
    save_log(testing_t_statistic, '/Users/yemao/Downloads/CDAN/CDAN_ACA/testing_t_statistic.pkl')
    return training_s_statistic, testing_s_statistic, testing_t_statistic
   

if __name__ == "__main__":
    training_s_statistic, testing_s_statistic, testing_t_statistic = main()
    plot_learning_curves(training_s_statistic, testing_s_statistic, testing_t_statistic)

