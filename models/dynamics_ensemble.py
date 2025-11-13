import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class DynNet(nn.Module):
    def __init__(self, in_dim, out_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class Ensemble:
    def __init__(self, in_dim, out_dim, M=5, lr=1e-3, device='cpu'):
        self.members = [DynNet(in_dim, out_dim).to(device) for _ in range(M)]
        self.opt = torch.optim.Adam([p for m in self.members for p in m.parameters()], lr=lr)
        self.device = device
    
    def mean_std(self, x):
        preds = torch.stack([m(x) for m in self.members], dim=0)
        return preds.mean(0), preds.std(0).clamp_min(1e-6)

    def train_epoch(self, X, Y, batch=1024):
        idx = np.random.permutation(len(X))
        losses = []

        for i in range(0, len(X), batch):
            sl = idx[i:i+batch]
            xb = torch.tensor(X[sl], dtype=torch.float32, device=self.device)
            yb = torch.tensor(Y[sl], dtype=torch.float32, device=self.device)
            loss = 0.0

            for m in self.members:
                loss = loss + F.mse_loss(m(xb), yb)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.setp()
            losses.append((loss.item())/len(self.members))

        return float(np.mean(losses))
    


