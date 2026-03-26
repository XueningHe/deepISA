import torch
from torch import nn
import torch.nn.functional as F




class Conv(nn.Module):
    def __init__(self, 
                 seq_len=600, 
                 ks=[15,9,9,9,9], 
                 cs=[64,64,64,64,64], 
                 ds=[1,2,4,8,16], 
                 dropout_rate=0.1):    
        super().__init__()
        if not (len(ks) == len(cs) == len(ds)):
            raise ValueError("Lengths of ks, cs, and ds must be equal.")
        self.seq_len = seq_len
        self.ks = ks
        self.ds = ds
        l = seq_len
        for i in range(len(ks)):
            l = l - ds[i] * (ks[i] - 1)
        if l <= 0:
            raise ValueError("Convolutional layers reduce output length to <= 0. check kernels/dilations.")
        self.flat_features = l * cs[-1]
        # 1. Convolutional backbone
        self.convs = nn.ModuleList([
            nn.Conv1d(4 if i==0 else cs[i-1], cs[i], kernel_size=ks[i], dilation=ds[i])
            for i in range(len(ks))
        ])
        # 2. Spatial Dropout (zeros entire channels)
        self.spatial_dropout = nn.ModuleList([
            nn.Dropout1d(p=dropout_rate) for _ in range(len(ks))
        ])
        # 3. Final Dropout (for the flattened features)
        self.final_dropout = nn.Dropout(p=dropout_rate)
        # 4. Task Heads
        self.regression_head = nn.Linear(self.flat_features, 1)
        self.classification_head = nn.Linear(self.flat_features, 1)
        self.rf = self._calculate_receptive_field()
        
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
        for conv, drop in zip(self.convs, self.spatial_dropout):
            x = F.relu(conv(x))
            x = drop(x)
            
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Regularize the fully connected transition
        x = self.final_dropout(x)
        
        reg = self.regression_head(x)     
        clf = self.classification_head(x)  
        return torch.cat([reg, clf], dim=1)