import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Model definition
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(Encoder, self).__init__()

        self.f1 = nn.Linear(x_dim, 512)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        # Additional hidden layer
        self.f2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.mu = nn.Linear(256, z_dim)
        self.log_sigma = nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.f1(x))))

        h = self.dropout2(self.bn2(self.act2(self.f2(h))))
        
        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10, 10)

        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, mod_dim, z_dim, emd_dim, num_batch):
        super(Decoder, self).__init__()

        self.alpha_mod = nn.Parameter(torch.randn(mod_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.batch_bias = nn.Parameter(torch.randn(num_batch, mod_dim))
        self.Topic_mod = None

    def forward(self, theta, batch_indices, cross_prediction=False):
        self.Topic_mod = torch.mm(self.alpha_mod, self.beta.t()).t()

        recon_mod = torch.mm(theta, self.Topic_mod)
        recon_mod += self.batch_bias[batch_indices]
        if not cross_prediction:
            recon_log_mod = F.log_softmax(recon_mod, dim=-1)
        else:
            recon_log_mod = F.softmax(recon_mod, dim=-1)

        return recon_log_mod

def MRF_ETM(input_dim, num_batch, num_topic=50, emd_dim=400):
    encoder = Encoder(x_dim=input_dim, z_dim=num_topic).cuda()
    decoder = Decoder(mod_dim=input_dim, z_dim=num_topic, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder.parameters()},
            {'params': decoder.parameters()}]

    optimizer = optim.Adam(PARA, lr=0.01)

    return encoder, decoder, optimizer