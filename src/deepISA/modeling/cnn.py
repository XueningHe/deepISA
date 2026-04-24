import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, 
                 mode="dual", 
                 model_config=None):    
        super().__init__()
        
        if model_config is None:
            model_config = {
                "seq_len": 600,
                "ks": [15,9,9,9,9], # kernel size
                "cs": [64,64,64,64,64], # channel size
                "ds": [1,2,4,8,16],
                "dropout": 0.1
            }

        self.mode=mode
        self.seq_len = model_config['seq_len']
        self.ks = model_config['ks']
        self.cs = model_config['cs']
        self.ds = model_config['ds']
        if not (len(self.ks) == len(self.cs) == len(self.ds)):
            raise ValueError("Lengths of ks, cs, and ds must be equal.")
        
        l = self.seq_len
        for i in range(len(self.ks)):
            l = l - self.ds[i] * (self.ks[i] - 1)
        if l <= 0:
            raise ValueError("Convolutional layers reduce output length to <= 0. check kernels/dilations.")
        flat_features = l * self.cs[-1]
        self.rf = self._calculate_receptive_field()
        # 1. Convolutional backbone
        self.convs = nn.ModuleList([
            nn.Conv1d(4 if i==0 else self.cs[i-1], self.cs[i], kernel_size=self.ks[i], dilation=self.ds[i])
            for i in range(len(self.ks))
        ])
        # 2. dropout layers
        self.dropout = nn.Dropout(p=getattr(model_config, 'dropout', 0.1))
        # 3. Task Heads
        self.regression_head = nn.Linear(flat_features, 1) 
        self.classification_head = nn.Linear(flat_features, 1) if mode=='dual' else None
        


    def _calculate_receptive_field(self):
        """
        Calculates the receptive field of a CNN with dilations.f
        RF = 1 + sum((kernel_size_i - 1) * dilation_i)
        """
        rf = 1
        for k, d in zip(self.ks, self.ds):
            rf += (k - 1) * d
        return rf


    def forward(self, x): 
        # Feature extraction
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.dropout(x)
        # Flatten
        x = x.view(x.size(0), -1) 
        # fc
        results = []
        if self.regression_head:
            results.append(self.regression_head(x))
        if self.classification_head:
            results.append(self.classification_head(x))
        return torch.cat(results, dim=1) if len(results) > 1 else results[0]