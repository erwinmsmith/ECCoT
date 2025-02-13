import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Trainer class
class Trainer_MRF_ETM(object):
    def __init__(self, encoder, decoder, optimizer):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder = None

    def train(self, x_mod, batch_indices, KL_weight, edge_links=None):
        self.toogle_grad(self.encoder, True)
        self.toogle_grad(self.decoder, True)

        self.encoder.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu, log_sigma = self.encoder(x_mod)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod.shape[0], mu.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

        recon_log_mod = self.decoder(Theta, batch_indices)

        nll_mod = (-recon_log_mod * x_mod).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        
        Loss = nll_mod + KL_weight * KL
        
        device = x_mod.device  # 获取当前设备 (CPU 或 CUDA)
        sim_loss = torch.tensor(0.0, device=device)  # 创建一个位于相同设备上的0张量
        if edge_links is not None:
            sim_loss = self.topic_similarity_loss(edge_links)
            Loss += sim_loss
    
        Loss.backward()
    
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)
    
        self.optimizer.step()
    
        return Loss.item(), nll_mod.item(), KL.item(), sim_loss.item()

    def validate(self, x_mod, batch_indices, KL_weight, edge_links=None):
        self.toogle_grad(self.encoder, False)
        self.toogle_grad(self.decoder, False)

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu, log_sigma = self.encoder(x_mod)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod.shape[0], mu.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod = self.decoder(Theta, batch_indices)

            nll_mod = (-recon_log_mod * x_mod).sum(-1).mean()

            KL = self.get_kl(mu, log_sigma).mean()
            
            Loss = nll_mod + KL_weight * KL
            
            device = x_mod.device  # 获取当前设备 (CPU 或 CUDA)
            sim_loss = torch.tensor(0.0, device=device)  # 创建一个位于相同设备上的0张量
            if edge_links is not None:
                sim_loss = self.topic_similarity_loss(edge_links)
                Loss += sim_loss
        
            return Loss.item(), nll_mod.item(), KL.item(), sim_loss.item()
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """
        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)
        
    def get_embed(self, x_mod):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu, log_sigma = self.encoder(x_mod)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod.shape[0], mu.shape[1]), use_cuda=x_mod.is_cuda)

            Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = mu.cpu().numpy()
        return out    
    
    def get_embed_best(self, x_mod):
        self.best_encoder.eval()

        with torch.no_grad():
            mu, log_sigma = self.best_encoder(x_mod)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod.shape[0], mu.shape[1]), use_cuda=x_mod.is_cuda)

            Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = mu.cpu().numpy()
        return out

    def get_NLL(self, x_mod, batch_indices):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu, log_sigma = self.encoder(x_mod)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod.shape[0], mu.shape[1]), use_cuda=x_mod.is_cuda)

            Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod = self.decoder(Theta, batch_indices)

            nll_mod = (-recon_log_mod * x_mod).sum(-1).mean()

        return nll_mod.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = torch.zeros(size)
        logvar = torch.zeros(size)
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2 * logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5 * torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

    def toogle_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def topic_similarity_loss(self, edge_links):
        Topic_mod = self.decoder.Topic_mod
        loss = 0.0
        num_pairs = 0
        for i, j in edge_links:
            col_i = Topic_mod[:, i]
            col_j = Topic_mod[:, j]
            if col_i.size(0) > 1 and col_j.size(0) > 1:
                cos_sim = F.cosine_similarity(col_i.unsqueeze(0), col_j.unsqueeze(0)).squeeze()
                loss += (cos_sim - 1.0) ** 2  # Assuming all connected pairs have a similarity of 1.0
                num_pairs += 1
        if num_pairs == 0:
            return 0.0
        return loss.mean()