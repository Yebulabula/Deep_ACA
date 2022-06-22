import torch 
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.data = torch.unsqueeze(self.data, -1)
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]