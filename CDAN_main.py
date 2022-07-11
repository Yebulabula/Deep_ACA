from __future__ import division
import torch
from torch.autograd import Variable
from CDAN.train import train
from CDAN.test import test
from CDAN.utils import plot_learning_curves, plot_cm, save_log
from data_loader import DA_dataloader
from model import  AdversarialNetwork, baseNetwork
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import warnings
import argparse

def main(source_data, target_data):
    """
    This method puts all the modules together to train a neural network
    classifier using CDAN loss.

    Reference: https://arxiv.org/abs/1705.10667
    """
    parser = argparse.ArgumentParser(description="domain adaptation for deep ACA")

    parser.add_argument("--epochs", default=50, type=int,
                        help="number of training epochs")

    parser.add_argument("--batch_size", default=128, type=int,
                        help="batch size of source data")

    parser.add_argument("--num_classes", default=3, type=int,
                        help="no. classes in dataset (default 3)")

    parser.add_argument("--d_input", default=1, type=int,
                        help="number of features (default 1)")

    parser.add_argument("--h", default=10, type=int,
                        help="number of heads (default 10)")

    parser.add_argument("--N", default=2, type=int,
                        help="number of encoders (default 2)")

    parser.add_argument("--q", default=4, type= int,
                        help="query size of transformer encoder block (4)")

    parser.add_argument("--v", default=4, type= int,
                        help="value size of transformer encoder block (4)")

    parser.add_argument("--lr", default=2e-3, type= int,
                        help="learning rate of generator (default 2e-3)")

    parser.add_argument("--d_model", default=16, type= int,
                        help="embedding dimension of the encoder (default 16)")

    parser.add_argument("--att_size", default=15, type= int,
                        help="attention mask size (default 15)")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model hyperparameters (paper page 5)
    LEARNING_RATE = args.lr
    MOMENTUM = 0.9
    d_model = args.d_model #32  Lattent dim
    q = args.q  # Query size
    v = args.v # Value size
    h = args.h # Number of heads
    N = args.N  # Number of epncoder and decoder to stack
    attention_size = args.att_size  # Attention window size
    pe = "regular"  # Positional encoding
    chunk_mode = None
    d_input = args.d_input # the number of channel of each data.
    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = args.num_classes
    bottleneck_dim = 512

    source_loader, target_loader = DA_dataloader(source_data, target_data, batch_size, device)

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
    ad_net = AdversarialNetwork(bottleneck_dim * num_classes, 1024)

    model.train(True)
    ad_net.train(True)

    # define optimizer pytorch: https://pytorch.org/docs/stable/optim.html
    # specify learning rates per layers:
    # 10*learning_rate for last two fc layers according to paper
    optimizer = torch.optim.SGD([
        {"params": model.sharedNetwork.parameters()},
        {"params": model.fc8.parameters(), "lr": LEARNING_RATE},
        {"params":ad_net.parameters(), "lr_mult": 15, 'decay_mult': 2}
    ], lr= LEARNING_RATE, momentum = MOMENTUM)

    # move to CUDA if available
    model = model.to(device)
    ad_net = ad_net.to(device)

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

        # run batch trainig at each epoch (returns dictionary with epoch result)
        result_train = train(model, ad_net, source_loader, target_loader,
                             optimizer, epoch+1, lambda_factor, device)
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
        test_source = test(model, source_loader, epoch, device)
        test_target = test(model, target_loader, epoch, device)

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
            torch.save(model, '{0}/CDAN_ACA_Exp_DA(4).pth'.format('CDAN/CDAN_ACA/model'))  

    Activities =  ['imp','ndm', 'oxa48']

    model = torch.load('{0}/CDAN_ACA_Exp_DA(4).pth'.format('CDAN/CDAN_ACA/model'))
    model.eval()

    true = pd.DataFrame(np.empty((len(target_loader.dataset))), columns= ['col1'])
    preds = pd.DataFrame(np.empty((len(target_loader.dataset))), columns= ['col1'])

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
    save_log(training_s_statistic, 'CDAN/CDAN_ACA/training_s_statistic.pkl')
    save_log(testing_s_statistic, 'CDAN/CDAN_ACA/testing_s_statistic.pkl')
    save_log(testing_t_statistic, 'CDAN/CDAN_ACA/testing_t_statistic.pkl')
    return training_s_statistic, testing_s_statistic, testing_t_statistic
   

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    with open('/root/example/Deep_ACA/dataset/PCR_clinical_isolates_data/gBlock_bs_lc_df_ac.csv', mode='r') as gblock:
        gblock_data = pd.read_csv(gblock)
        
    with open('/root/example/Deep_ACA/dataset/PCR_clinical_isolates_data/new_clinical_df_ac.csv', mode='r') as clinical:
        clinical_data = pd.read_csv(clinical)

    source_data = gblock_data[(~gblock_data["Target"].isin(['kpc', 'vim']))]
    target_data = clinical_data[(~clinical_data["Target"].isin(['kpc', 'vim', 'ndm_oxa48']))]

    training_s_statistic, testing_s_statistic, testing_t_statistic = main(source_data, target_data)
    plot_learning_curves(training_s_statistic, testing_s_statistic, testing_t_statistic)