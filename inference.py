import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def inference(trainer, new_data, batch_indices):
    trainer.encoder.eval()
    trainer.decoder.eval()

    with torch.no_grad():
        new_data = new_data.to('cuda')
        batch_indices = batch_indices.to('cuda')

        mu, log_sigma = trainer.encoder(new_data)
        mu_prior, logsigma_prior = trainer.prior_expert((1, new_data.shape[0], mu.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma.unsqueeze(0)), dim=0)

        mu, log_sigma = trainer.experts(Mu, Log_sigma)

        Theta = F.softmax(trainer.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

        recon_log_mod = trainer.decoder(Theta, batch_indices)

        # Extracting required matrices
        topic_embeddings = trainer.decoder.beta.detach().cpu().numpy()
        word_embeddings = trainer.decoder.alpha_mod.detach().cpu().numpy()
        topic_word_distribution = F.softmax(trainer.decoder.Topic_mod, dim=-1).detach().cpu().numpy()

    return Theta.cpu().numpy(), recon_log_mod.cpu().numpy(), topic_embeddings, word_embeddings, topic_word_distribution