import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_npu
from torch.utils.data import DataLoader
from models.unet_v1 import UNet
from models.ddpm import DDPM
# from models.simplenet import UNet
# from models.ddpm_nocond import DDPM_nocond as DDPM
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.functional as F
import warnings
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# 忽略torch_npu的警告
warnings.filterwarnings("ignore", message="AutoNonVariableTypeMode is deprecated")
from data.dataset_diff2 import get_dataloaders,get_dataloaders_combine
# 混合精度
from torch.cuda.amp import autocast, GradScaler

# 配置参数
RESUME = False
NUM_EPOCH = 100
device = "npu:0"
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-6
ACCUMULATION_STEPS = 2  # 梯度累积步数
save_dir = "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv4_unetv1"
data_path = "/home/docker/data/private/AuroraData/combined_data/hourly/combined_all_years_hourly.npy"

# save_dir = "/home/docker/code/Aurora_DDPM/ckpt/diffusion_ckpt_simplenet/ckpt_v7"
os.makedirs(save_dir, exist_ok=True)
# 创建模型
unet = UNet(1, 1)
ddpm = DDPM(unet, num_train_steps=1000, schedule='cosine')
optimizer = optim.AdamW(ddpm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ddpm.to(device)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# 加载数据
years = [1996]
months = range(4, 6)
# train_loader, val_loader = get_dataloaders_combine(
#     data_path=data_path,
#     batch_size=BATCH_SIZE,
#     train_ratio=0.8,
#     save_norm_params=True,
#     norm_params_path=os.path.join(save_dir,'norm_params.pkl')
# )
train_loader, val_loader = get_dataloaders(
    years=years,
    months=months,
    batch_size=BATCH_SIZE,
    train_ratio=0.9,
    save_norm_params=True,
    norm_params_path=os.path.join(save_dir,'norm_params.pkl')
)
# 训练记录
train_losses = []  # 训练损失（每个batch）
val_losses = []    # 验证损失（每个epoch）
epoch_train_losses = []  # 每个epoch的平均训练损失

# 保存路径
model_dir = "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv1_unetv1"
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "aurora_diff_best.pth")
checkpoint_path = os.path.join(model_dir, "checkpoint_epoch_85.pth")

def validate_model(model, val_loader, device):
    """在验证集上评估模型"""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (aurora_data, omni_data) in enumerate(val_loader):
            # 转移到设备
            aurora_data = aurora_data.to(device, dtype=torch.float32)
            omni_data = omni_data.to(device, dtype=torch.float32)
            
            # 随机时间步
            t = torch.randint(0, 1000, (aurora_data.size(0),)).long().to(device)
            
            # 混合精度前向传播（即使验证也使用混合精度以减少内存）
            with autocast():
                x_t, noise = ddpm.add_noise(aurora_data, t)
                # 前向传播
                pred_noise = ddpm(x_t, t, omni_data)
                loss = F.mse_loss(pred_noise, noise)
            
            total_val_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Validation batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.6f}")
    
    avg_val_loss = total_val_loss / max(num_batches, 1)
    return avg_val_loss

def train_epoch(model, train_loader, optimizer, scaler, device, accumulation_steps=2):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    current_batch = 0
    
    for batch_idx, (aurora_data, omni_data) in enumerate(train_loader):
        # 转移到设备
        aurora_data = aurora_data.to(device, dtype=torch.float32)
        omni_data = omni_data.to(device, dtype=torch.float32)
        # 随机时间步
        t = torch.randint(0, 1000, (aurora_data.size(0),)).long().to(device)
        
        # 混合精度训练
        with autocast():
            x_t, noise = ddpm.add_noise(aurora_data, t)
            # 前向传播
            pred_noise = ddpm(x_t, t, omni_data)
            loss = F.mse_loss(pred_noise, noise)
        
        # 梯度累积
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        current_batch += 1
        if current_batch % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_batch = 0
        
        # 记录损失（乘以累积步数以获得实际损失）
        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss
        num_batches += 1
        train_losses.append(batch_loss)
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx:04d}/{len(train_loader):04d}, Loss: {batch_loss:.6f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def plot_losses(train_losses_per_epoch, val_losses_per_epoch, save_path):
    """绘制训练和验证损失曲线"""
    epochs = list(range(1, len(train_losses_per_epoch) + 1))
    
    plt.figure(figsize=(15, 5))
    
    # 子图1：训练和验证损失（每个epoch）
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses_per_epoch, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses_per_epoch, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：损失下降趋势（对数尺度）
    plt.subplot(1, 3, 2)
    plt.semilogy(epochs, train_losses_per_epoch, 'b-', label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_losses_per_epoch, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Trends (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：移动平均的训练损失（每个batch）
    plt.subplot(1, 3, 3)
    if len(train_losses) > 0:
        # 计算移动平均
        window_size = min(100, len(train_losses) // 10)
        if window_size > 1:
            moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size, len(train_losses)+1), moving_avg, 'g-', 
                    label=f'Moving Avg (window={window_size})', linewidth=2)
        plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', alpha=0.3, label='Raw Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Batch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")

def main():
    # 初始化混合精度训练
    scaler = GradScaler()
    print("启用混合精度训练")
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if RESUME and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ddpm.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses.extend(checkpoint.get('train_losses', []))
        val_losses.extend(checkpoint.get('val_losses', []))
        epoch_train_losses.extend(checkpoint.get('epoch_train_losses', []))
        print(f"恢复训练: 从epoch {start_epoch}开始, 最佳验证损失: {best_val_loss:.6f}")
    
    # 训练循环
    for epoch in range(start_epoch, NUM_EPOCH):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCH}")
        print(f"{'='*60}")
        
        # 训练阶段
        print("Training phase...")
        train_loss = train_epoch(ddpm, train_loader, optimizer, scaler, device, ACCUMULATION_STEPS)
        epoch_train_losses.append(train_loss)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证阶段
        print("Validation phase...")
        val_loss = validate_model(ddpm, val_loader, device)
        val_losses.append(val_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # 打印epoch统计信息
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training Loss: {train_loss:.6f}")
        print(f"  Validation Loss: {val_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch_train_losses': epoch_train_losses,
            }, best_model_path)
            print(f"  ✓ 保存最佳模型! 验证损失: {val_loss:.6f}")
        
        # 每5个epoch保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch_train_losses': epoch_train_losses,
            }, checkpoint_save_path)
            print(f"  ✓ 保存检查点到 {checkpoint_save_path}")
        
        # 绘制损失曲线
        plot_losses(epoch_train_losses, val_losses, os.path.join(save_dir, "loss_curves.png"))
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, "aurora_diff_final.pth")
    torch.save({
        'epoch': NUM_EPOCH,
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epoch_train_losses': epoch_train_losses,
    }, final_model_path)
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最终模型已保存到 {final_model_path}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 绘制最终的训练摘要
    plt.figure(figsize=(15, 10))
    
    # 子图1：训练和验证损失对比
    plt.subplot(2, 2, 1)
    epochs = list(range(1, len(epoch_train_losses) + 1))
    plt.plot(epochs, epoch_train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：损失对数图
    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, epoch_train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.semilogy(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Trends (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：训练损失分布（箱线图）
    plt.subplot(2, 2, 3)
    if len(train_losses) > 0:
        # 将训练损失分成几段用于箱线图
        num_segments = min(10, len(epochs))
        segment_size = len(train_losses) // num_segments
        loss_segments = []
        segment_labels = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(train_losses)
            loss_segments.append(train_losses[start_idx:end_idx])
            segment_labels.append(f'Epoch {(i+1)*len(epochs)//num_segments}')
        
        plt.boxplot(loss_segments, labels=segment_labels)
        plt.xlabel('Training Progress')
        plt.ylabel('Loss')
        plt.title('Training Loss Distribution')
        plt.xticks(rotation=45)
    
    # 子图4：最佳验证损失标记
    plt.subplot(2, 2, 4)
    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.plot(epochs, val_losses, 'r-', linewidth=2)
    plt.scatter(best_epoch, min(val_losses), color='green', s=200, 
               label=f'Best: {min(val_losses):.6f}', zorder=5)
    plt.axhline(y=min(val_losses), color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'Best Validation Loss (Epoch {best_epoch})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_summary.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练摘要图已保存到 {os.path.join(save_dir, 'training_summary.png')}")

if __name__ == "__main__":
    main()