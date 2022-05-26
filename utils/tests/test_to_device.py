import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.training import to_device


class Dataset(Dataset):
    def __init__(self):
        self.data1 = pd.DataFrame(np.random.random((16, 3)))
        self.data2 = pd.DataFrame(np.random.random((16, 3)))
    
    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, idx):
        return (
            self.data1.iloc[idx, :].to_numpy(),
            self.data2.iloc[idx, :].to_numpy(),
        )

def test_base_dataloader():
    data = Dataset()

    loader1 = DataLoader(data, batch_size=4, shuffle=False)
    loader2 = DataLoader(data, batch_size=4, shuffle=False)

    for batch1, batch2 in zip(loader1, loader2):
        assert batch1[0].equal(batch2[0])
        assert batch1[1].equal(batch2[1])

    
def test_to_device():
    data = Dataset()

    loader1 = DataLoader(data, batch_size=4, shuffle=False)
    loader2 = DataLoader(data, batch_size=4, shuffle=False)

    device = torch.float32

    for (a1, a2), (b1, b2) in zip(loader1, to_device(loader2, device)):
        assert a1.dtype == torch.float64
        assert b1.dtype == torch.float32

        assert a1.to(torch.float32).equal(b1)
        assert a2.to(torch.float32).equal(b2)

    