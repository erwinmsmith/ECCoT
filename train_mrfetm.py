import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from utils import calc_weight

def Train_MRF_ETM(trainer, dataloader_train, dataloader_val, Total_epoch, edge_links):
    # 定义早停和学习率调度的参数
    patience = 20  # 如果验证损失在10个epoch内没有下降，则触发早停
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # 创建余弦退火学习率调度器
    T_max = Total_epoch  # 单周期的最大迭代次数
    eta_min = 1e-6  # 最小学习率
    scheduler = lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=T_max, eta_min=eta_min)

    for epoch in range(Total_epoch):
        if early_stop:
            print("Early stopping triggered")
            break
        
        trainer.encoder.train()
        trainer.decoder.train()

        Loss_all = 0
        NLL_all = 0
        KL_all = 0
        Sim_Loss_all = 0

        tstart = time.time()

        for batch_features, batch_indices in dataloader_train:
            batch_features = batch_features.to('cuda')
            batch_indices = batch_indices.to('cuda')

            KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-5)

            # 检查 edge_links 是否非空
            if edge_links and len(edge_links) > 0:
                loss, nll, kl, sim_loss = trainer.train(batch_features, batch_indices, KL_weight, edge_links=edge_links)
                Sim_Loss_all += ensure_tensor(sim_loss).item()  # 确保 sim_loss 是张量
            else:
                # 如果 edge_links 为空，则不传递该参数
                loss, nll, kl, _ = trainer.train(batch_features, batch_indices, KL_weight)

            Loss_all += ensure_tensor(loss).item()  # 确保 loss 是张量
            NLL_all += ensure_tensor(nll).item()  # 确保 nll 是张量
            KL_all += ensure_tensor(kl).item()  # 确保 kl 是张量

        # Validation step
        trainer.encoder.eval()
        trainer.decoder.eval()

        val_loss_all = 0
        val_nll_all = 0
        val_kl_all = 0
        val_sim_loss_all = 0

        with torch.no_grad():
            for val_features, val_indices in dataloader_val:
                val_features = val_features.to('cuda')
                val_indices = val_indices.to('cuda')

                KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-5)

                # 检查 edge_links 是否非空
                if edge_links and len(edge_links) > 0:
                    val_loss, val_nll, val_kl, val_sim_loss = trainer.validate(val_features, val_indices, KL_weight, edge_links=edge_links)
                    val_sim_loss_all += ensure_tensor(val_sim_loss).item()  # 确保 val_sim_loss 是张量
                else:
                    # 如果 edge_links 为空，则不传递该参数
                    val_loss, val_nll, val_kl, _ = trainer.validate(val_features, val_indices, KL_weight)

                val_loss_all += ensure_tensor(val_loss).item()  # 确保 val_loss 是张量
                val_nll_all += ensure_tensor(val_nll).item()  # 确保 val_nll 是张量
                val_kl_all += ensure_tensor(val_kl).item()  # 确保 val_kl 是张量

        tend = time.time()
        
        # 计算平均损失
        avg_train_loss = Loss_all / len(dataloader_train)
        avg_train_nll = NLL_all / len(dataloader_train)
        avg_train_kl = KL_all / len(dataloader_train)
        avg_train_sim_loss = Sim_Loss_all / len(dataloader_train) if edge_links and len(edge_links) > 0 else None
        avg_val_loss = val_loss_all / len(dataloader_val)
        avg_val_nll = val_nll_all / len(dataloader_val)
        avg_val_kl = val_kl_all / len(dataloader_val)
        avg_val_sim_loss = val_sim_loss_all / len(dataloader_val) if edge_links and len(edge_links) > 0 else None

        # 更新学习率
        scheduler.step()

        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 检查是否触发早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            trainer.best_encoder = trainer.encoder.state_dict().copy()  # 确保复制的是深拷贝
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                early_stop = True
                print(f'Validation loss has not improved for {patience} epochs. Stopping training.')

        # 打印信息
        print(f'Epoch={epoch}, Time={tend - tstart:.4f}, LR={current_lr:.8f}, '
              f'Train Loss={avg_train_loss:.4f}, Train NLL={avg_train_nll:.4f}, Train KL={avg_train_kl:.4f}, ' +
              (f'Train Sim_Loss={avg_train_sim_loss:.4f}, ' if avg_train_sim_loss is not None else '') +
              f'Val Loss={avg_val_loss:.4f}, Val NLL={avg_val_nll:.4f}, Val KL={avg_val_kl:.4f}, ' +
              (f'Val Sim_Loss={avg_val_sim_loss:.4f}' if avg_val_sim_loss is not None else ''))

    # 在训练结束后，你可以加载最佳的编码器状态
    if trainer.best_encoder is not None:
        trainer.encoder.load_state_dict(trainer.best_encoder)
        print("Loaded the best encoder state.")

    return trainer

# 辅助函数：确保输入是张量
def ensure_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, (int, float)):
        return torch.tensor(value, device='cuda')
    else:
        raise ValueError(f"Unsupported type: {type(value)}")