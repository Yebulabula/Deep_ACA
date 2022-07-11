import torch
from torch.autograd import Variable

def test(model, data_loader, epoch, device):
    """
        This method is used for evaluating the performance of model on the testing 
        target domain data.

        Implementation based on:
        https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/CDAN/test.py

        INPUT:
        model: Generator network.
        data_loader: target domain dataloader.
        epoch: the current epoch number for testing.
        device: choose cuda/cpu device to train models.

        RETURN:
        results: A dictionary containing test info of each epoch.
    """

    model.eval() # the model switches from training mode to the testing mode.

    test_loss = 0
    correct_class = 0

    # go over dataloader batches, labels
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)

        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
        data, label = Variable(data, volatile=True), Variable(label)
        _, output = model(data) # just use one ouput of DeepCORAL

        # sum batch loss when computing classification
        test_loss += torch.nn.functional.cross_entropy(output, label, size_average=False).item()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

    # compute test loss as correclty classified labels divided by total data size
    test_loss = test_loss/len(data_loader.dataset)

    return {
        "epoch": epoch,
        "average_loss": test_loss,
        "correct_class": correct_class,
        "total_elems": len(data_loader.dataset),
        "accuracy %": 100.*correct_class/len(data_loader.dataset)
    }