import pickle
import torch

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PNNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        with open(self.x[index], 'rb') as f:
            x = pickle.load(f)
            x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor([self.y[index]], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.x)

def make_dataloader(csv_path, batch_size = 128, shuffle = True, test_mode = False):
    df = pd.read_csv(csv_path, header=None)
    file_path_list, labels = df[0].values, df[1].values
    dataset = PNNDataset(file_path_list, labels)
    dataloader = DataLoader(dataset, batch_size, shuffle = shuffle)
    if test_mode:
        return dataloader, file_path_list
    else:
        return dataloader

def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def train_help():
    pass

def preprocess_help():
    pass