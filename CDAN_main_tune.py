from __future__ import division
import torch
from functools import partial
from CDAN.train import train
from CDAN.test import test
from data_loader import DA_dataloader
from model import  AdversarialNetwork, baseNetwork
import pandas as pd
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import warnings
import os

# set model hyperparameters (paper page 5)
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
d_model = 16 #32  Lattent dim
q = 4  # Query size
v = 4  # Value size
h = 10 #4   Number of heads
N = 2  # Number of epncoder and decoder to stack
attention_size = 15  # Attention window size
pe = "regular"  # Positional encoding
chunk_mode = None
d_input = 1 # the number of channel of each data.
epochs = 100
batch_size = 128
num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config, source_loader, target_loader, checkpoint_dir = None, data_dir = None):
    """
    This method puts all the modules together to train a neural network
    classifier using cdan loss. (Hyperparameter Tuning version)

    Reference: https://arxiv.org/abs/1705.10667
    """
    bottleneck_dim = 512

    # define generator and classifier network
    model = baseNetwork(d_input, 
                        config['d_model'], 
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
        {"params": model.fc8.parameters(), "lr": config["lr"]},
        {"params":ad_net.parameters(), "lr_mult": config["lr_mult"], 'decay_mult': 2}
    ], lr= LEARNING_RATE, momentum = MOMENTUM)

    # move to CUDA if available
    model = model.to(device)
    ad_net = ad_net.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        ad_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    print("model type:", type(model))

    # store statistics of train/test
    training_s_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    # start training over epochs
    print("running training for {} epochs...".format(epochs))

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), ad_net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss = test_target['average_loss'], accuracy = test_target['accuracy %'])

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
   

if __name__ == "__main__":
    data_dir = os.path.abspath("/root/example/Deep_ACA/model_ray")

    warnings.filterwarnings("ignore")
    
    with open('/root/example/Deep_ACA/dataset/PCR_clinical_isolates_data/gBlock_bs_lc_df_ac.csv', mode='r') as gblock:
        gblock_data = pd.read_csv(gblock)
        
    with open('/root/example/Deep_ACA/dataset/PCR_clinical_isolates_data/new_clinical_df_ac.csv', mode='r') as clinical:
        clinical_data = pd.read_csv(clinical)

    # gblock_data = pd.concat([gblock_data,gblock_data,gblock_data])
    source_data = gblock_data[(~gblock_data["Target"].isin(['kpc', 'vim']))]
    target_data = clinical_data[(~clinical_data["Target"].isin(['kpc', 'vim', 'ndm_oxa48']))]

    source_loader, target_loader = DA_dataloader(source_data, target_data, batch_size, device)

    config = {
        # self-defined sampling method.
        "d_model": tune.choice([16, 32, 24, 64]),
        # Random distributed sample.
        "lr": tune.loguniform(1e-3, 1e-2),
        "lr_mult": tune.choice([10, 15, 20]),
    }

    # ASHAScheduler early stopping the training based on critera.
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t= 50,
        grace_period=10,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    # Carry out training process
    result = tune.run(
        partial(main, source_loader = source_loader, target_loader = target_loader, data_dir=data_dir),
        # Specify training resources
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples= 20,
        scheduler=scheduler,
        progress_reporter=reporter)
 
    # Find the best trail.
    best_trial = result.get_best_trial("accuracy", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))