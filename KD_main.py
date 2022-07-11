import torch
import numpy as np
from model import Discriminators
import torch
import torch.optim as optim
import numpy as np
from models import *
import warnings 
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loader import dataset
from model import baseNetwork
from CDAN.utils import _str_to_int
from KD.train import train
from KD.test import test

# ================= Arugments ================ #

# Knowledge Distillation Hyperparameters
lr = 1e-3
d_lr = 5e-2
gpu_id = '0'
gamma = '[1,1,1,1,1]'
eta = '[1,1,1,1,1]'
loss = "ce"
out_layer = "[-1]"
epochs = 100
batch_size = 128
base_lr = 0.2
weight_decay = 1e-4
momentum = 0.9
nesterov = True
grl = False
lr_min = 0
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Transformer Model Hyperparameters
d_model = 16 #32  Lattent dim
q = 4  # Query size
v = 4  # Value size
h = 2 #4   Number of heads
N = 1  # Number of encoder and decoder to stack
attention_size = 15  # Attention window size
pe = "regular"  # Positional encoding
chunk_mode = None
d_input = 1 # the number of channel of each data.
num_classes = 7
drop_out =0.3


if __name__ == "__main__":
    device = 'cpu'
    warnings.filterwarnings("ignore")

    # ================= Dataloader Setup ================ #
    with open('/Users/yemao/Documents/Deep_ACA/dataset/train_X.npy', 'rb') as f:
            target_data = np.load(f, allow_pickle=True)


    with open('/Users/yemao/Documents/Deep_ACA/dataset/train_y.npy', 'rb') as f:
            target_label = np.load(f, allow_pickle=True)
            target_label = _str_to_int(target_label)

    with open('/Users/yemao/Documents/Deep_ACA/dataset/test_X.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)
        test_data = test_data.astype(np.float32)

    with open('/Users/yemao/Documents/Deep_ACA/dataset/test_y.npy', 'rb') as f:
            test_label = np.load(f, allow_pickle=True)
            test_label = _str_to_int(test_label)

    target_data = target_data.astype(np.float32)
    
    target_dataset = dataset(torch.from_numpy(target_data).to(device), torch.from_numpy(target_label).to(device))


    test_data, valid_data, test_label, valid_label  = train_test_split(test_data, test_label, test_size=0.5, random_state=42, shuffle=True)
    print(target_data.shape, test_data.shape, valid_data.shape, batch_size)
    test_dataset = dataset(torch.from_numpy(test_data).to(device), torch.from_numpy(test_label).to(device))
    valid_dataset = dataset(torch.from_numpy(valid_data).to(device), torch.from_numpy(valid_label).to(device))

    target_loader = DataLoader(dataset = target_dataset, batch_size = batch_size, shuffle = False)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    # ================= Teacher Setup ================ #
    teachers = []
    model_name = ['1', '2']
    for i in model_name:
        model = torch.load('/Users/yemao/Documents/Deep_ACA/CDAN/CDAN_ACA/model/CDAN_ACA_Exp_DA({}).pth'.format(i)).to(device)
        model.eval()
        teachers.append(model)

    # ================= Student Setup ================ #
    bottleneck_dim = 256
    student = baseNetwork(d_input, 
                            d_model, 
                            q, v, h, N, 
                            attention_size, 
                            chunk_mode,
                            pe, 
                            pe_period=45,
                            num_classes= num_classes,
                            bottleneck_dim = bottleneck_dim).to(device)

    # ================= Discriminator Setup ================ #
    dims = [256, 7]
    print("dims:", dims)
    discriminators = Discriminators(dims, grl= grl)

    # ================= Optimizer Setup ================ #
    update_parameters = [{'params': student.parameters()}]
    
    best_acc = 0
    patience = 0
    for d in discriminators.discriminators:
        d = d.to(device)
        if device == "cuda":
            d = torch.nn.DataParallel(d)
        update_parameters.append({'params': d.parameters(), "lr": d_lr})

    optimizer = optim.SGD(update_parameters, lr = lr, momentum = momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 * min(1, len(teachers)), 250 * min(1, (len(teachers)))],gamma=0.1)

    for epoch in range(start_epoch, start_epoch + 2):
        lambda_factor = lambda_factor = (epoch+1)/epochs
        valid_acc = train(teachers, student, discriminators, epoch, target_loader, valid_loader, optimizer, scheduler, lambda_factor)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(student, '{0}/Deep_ACA_KD.pth'.format('/Users/yemao/Documents/Deep_ACA/CDAN/CDAN_ACA/model'))
            patience = 0
        patience += 1
        # if patience > 20:
        #     break

    print()
    print()
    print("##### Testing #####")
    best_model = torch.load('{0}/Deep_ACA_KD.pth'.format('/Users/yemao/Documents/Deep_ACA/CDAN/CDAN_ACA/model'))
    test_acc =  test(teachers, best_model, test_loader)
    print(test_acc)