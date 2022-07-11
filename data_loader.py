import torch 
from torch.utils.data import Dataset, DataLoader
from CDAN.utils import  split_data_labels, str_to_int, df_to_tensor

class dataset(Dataset):
    """
        This class is used to encapsulate data and labels into a dataset.
    """
    def __init__(self, data, labels):
        self.data = data
        self.data = torch.unsqueeze(self.data, -1)
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def DA_dataloader(source_data, target_data, batch_size, device):
    """
        This function is used to create dataloader with the given batch size for source and target data.

        INPUT:
        source_data: g-block dataframes.
        target_data: clinical isolates dataframes.
        batch_size: the number of data feed into the model for training at each iteration.
        device: check if using cpu/cuda.

        RETURN:
        source_loader: source domain dataloader for training.
        target_loader: target domain dataloader for training.
    """

    source_X_df, source_Y_df = split_data_labels(source_data)
    target_X_df, target_Y_df = split_data_labels(target_data)

    source_X, source_Y = df_to_tensor(source_X_df),  torch.from_numpy(str_to_int(source_Y_df))
    target_X, target_Y = df_to_tensor(target_X_df),  torch.from_numpy(str_to_int(target_Y_df))

    # create dataloaders (gblock DNA as source, clinical isolates as target)
    source_dataset = dataset(source_X.to(device), source_Y.to(device))
    target_dataset = dataset(target_X.to(device), target_Y.to(device))

    source_loader = DataLoader(dataset = source_dataset, batch_size = batch_size, shuffle = False)
    target_loader = DataLoader(dataset = target_dataset, batch_size = batch_size, shuffle = False)
    return source_loader, target_loader
