import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import warnings
import torch
import torch.nn as nn
import torch_npu
from models.unet_v3 import UNet
from models.ddpm import DDPM
from datetime import datetime, timedelta
from utils.normlize import normalize, denormalize
from data.dataset_diff import OmniDataset
from models.simplenet import UNet as UNet_nocond
from models.ddpm_nocond import DDPM_nocond
warnings.filterwarnings('ignore')

# ==================== 1. 参数设置 ====================
mask_size = (20, 20)  # 掩码大小 (高度, 宽度)
results_dir = "/home/docker/code/Aurora_DDPM/reasult/eval_res/new_res/eval_results"
os.makedirs(results_dir, exist_ok=True)
num_steps = 300
repaint_steps = 10
jump_len = 10
n_samples = 1
device = "npu:0"
model_save_path = "/home/docker/code/Aurora_DDPM/ckpt/diffusion_ckpt_unet/ckpt_v2_unetv3/aurora_diff_best.pth"

# ==================== 2. 加载模型 ====================
print("加载扩散模型...")
unet = UNet(1, 1)
ddpm = DDPM(unet, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path, map_location=device)
ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
ddpm.eval()
ddpm.to(device)

model_save_path_nocond = "/home/docker/code/Aurora_DDPM/ckpt/diffusion_ckpt_simplenet/ckpt_v8/aurora_diff_best.pth"
unet_nocond = UNet_nocond(1, 1)
ddpm_nocond = DDPM_nocond(unet_nocond, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path_nocond, map_location=device)
ddpm_nocond.load_state_dict(checkpoint['model_state_dict'], strict=False)
ddpm_nocond.eval()
ddpm_nocond.to(device)

# ==================== 3. 加载测试数据 ====================
print("加载测试数据...")

data1_path = "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/aurora_img_20050101.npy"
mn_datas = np.load(data1_path)
omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_1min_pro/2005/omni_20050101_1min.npy"
omni_data = np.load(omni_path)
mn_time = omni_data['utc']

solar_fields = ['Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'P']
solar_components = []
for field in solar_fields:
    if field in omni_data.dtype.names:
        field_data = omni_data[field]
        if field_data.ndim > 1:
            if field_data.shape[1] > 0:
                field_data = field_data[:, 0]
            else:
                field_data = field_data.flatten()
        solar_components.append(field_data.astype(np.float32))
    else:
        print(f"字段 {field} 不存在")
solar_data = np.column_stack(solar_components)
solar_data = OmniDataset(solar_data)

# 一个月的数据，每分钟一个点，总共约30*24*60=43200个点
# 每隔6小时（360分钟）取一个点
total_data_points = len(mn_datas)
#total_data_points = 90
interval = 360  # 6小时 = 360分钟
selected_indices = list(range(0, total_data_points, interval))

# 确保不超过数据范围
selected_indices = selected_indices[:min(len(selected_indices), 240)]  # 一个月最多240个点（30天*8个点/天）

# 选择数据
mn_timestamps = mn_time[selected_indices]
mn_data = mn_datas[selected_indices]

normalizer_real = normalize(mn_datas)
denormalizer_real = denormalize(mn_datas)
print(f"加载了 {len(selected_indices)} 个测试时间点")
print(f"图像形状: {mn_datas.shape}")

# ==================== 4. 辅助函数 ====================
def convert_datetime64_to_datetime(dt64):
    """将 numpy.datetime64 转换为 datetime.datetime"""
    if dt64 is None:
        return None
    if isinstance(dt64, datetime):
        return dt64
    import pandas as pd
    return pd.Timestamp(dt64).to_pydatetime()

def interpolate_inpainting(corrupted_img, mask):
    """使用双线性插值进行修复"""
    known_coords = np.column_stack(np.where(mask == 1))
    unknown_coords = np.column_stack(np.where(mask == 0))
    
    if len(known_coords) == 0 or len(unknown_coords) == 0:
        return corrupted_img.copy()
    
    known_values = corrupted_img[mask == 1]
    interpolated_values = griddata(
        known_coords, 
        known_values, 
        unknown_coords, 
        method='linear',
        fill_value=0
    )
    
    restored_img = corrupted_img.copy()
    restored_img[mask == 0] = interpolated_values
    
    return restored_img

def ddpm_repair(input_data, mask, solar_point):
    """扩散模型修复函数（条件）"""
    input_data_norm = normalizer_real(input_data)
    input_data_norm = np.expand_dims(input_data_norm, axis=(0,1))
    mask_expanded = np.expand_dims(mask, axis=(0, 1))
    
    input_tensor = torch.tensor(input_data_norm).float().to(device)
    mask_tensor = torch.tensor(mask_expanded).float().to(device)
    solar_tensor = solar_point.unsqueeze(0).to(device)
    
    with torch.no_grad():
        repaired = ddpm.sample(
            input_tensor,
            mask_tensor,
            solar_tensor,
            num_inference_steps=num_steps,
            n_sample=n_samples,
            j=jump_len,
            r=repaint_steps,
        )
    
    repaired_np = repaired.cpu().numpy().squeeze()
    repaired_flux = denormalizer_real(repaired_np)
    return repaired_flux

def ddpm_repair_nocond(input_data, mask):
    """扩散模型修复函数（无条件）"""
    input_data_norm = normalizer_real(input_data)
    input_data_norm = np.expand_dims(input_data_norm, axis=(0,1))
    mask_expanded = np.expand_dims(mask, axis=(0, 1))
    
    input_tensor = torch.tensor(input_data_norm).float().to(device)
    mask_tensor = torch.tensor(mask_expanded).float().to(device)
    
    with torch.no_grad():
        repaired = ddpm_nocond.sample(
            input_tensor,
            mask_tensor,
            num_inference_steps=num_steps,
            n_sample=n_samples,
            j=jump_len,
            r=repaint_steps,
        )
    
    repaired_np = repaired.cpu().numpy().squeeze()
    repaired_flux = denormalizer_real(repaired_np)
    return repaired_flux

def calculate_ssim(original, restored, mask):
    """计算SSIM (结构相似性)"""
    # 提取缺失区域
    img1_masked = original[mask == 0].flatten()
    img2_masked = restored[mask == 0].flatten()
    
    if len(img1_masked) == 0:
        return 1.0
    
    # 计算均值
    mu1 = np.mean(img1_masked)
    mu2 = np.mean(img2_masked)
    
    # 计算方差和协方差
    sigma1_sq = np.var(img1_masked)
    sigma2_sq = np.var(img2_masked)
    sigma12 = np.cov(img1_masked, img2_masked)[0, 1]
    
    # SSIM参数
    C1 = (0.01 * np.max([np.max(original), np.max(restored)])) ** 2
    C2 = (0.03 * np.max([np.max(original), np.max(restored)])) ** 2
    
    # 计算SSIM
    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return ssim_numerator / ssim_denominator

def calculate_psnr(original, restored, mask):
    """计算PSNR (峰值信噪比)"""
    # 只计算掩码区域的MSE
    mse = np.mean(((original - restored) ** 2)[mask == 0])
    if mse == 0:
        return float('inf')
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mae(original, restored, mask):
    """计算MAE (平均绝对误差)"""
    # 只计算掩码区域
    mae = np.mean(np.abs(original - restored)[mask == 0])
    return mae

def calculate_rmse(original, restored, mask):
    """计算RMSE (均方根误差)"""
    # 只计算掩码区域
    mse = np.mean(((original - restored) ** 2)[mask == 0])
    rmse = np.sqrt(mse)
    return rmse

def calculate_r2(original, restored, mask):
    """计算决定系数R²（相关系数）"""
    # 提取掩码区域
    y_true = original[mask == 0].flatten()
    y_pred = restored[mask == 0].flatten()
    
    if len(y_true) == 0:
        return 1.0
    
    # 计算残差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 计算总平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 避免除以零
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    # 计算R²
    r2 = 1 - (ss_res / ss_tot)
    
    # R²可能为负（模型比平均值还差），我们限制在合理范围内
    return max(r2, -1.0)

def create_mask_in_mlt_region(image_shape, mlat_range=(65, 75)):
    """
    在指定的MLT区域创建掩码
    
    Args:
        image_shape: 图像形状 (80, 96)
        mlat_range: 纬度范围，默认为65-75度
    
    Returns:
        mask: 掩码矩阵，1表示保留，0表示挖去
    """
    h, w = image_shape
    # MLT转换为列索引 (96列对应0-24小时)
    # 每列对应的MLT小时：col_idx * 24 / 96
    
    mlt_start1 = 18
    mlt_end1 = 24
    
    mlt_start2 = 0
    mlt_end2 = 6
    
    # 计算列索引范围
    col_start = int(mlt_start1 * w / 24)
    col_end = int(mlt_end1 * w / 24)
    
    col_start2 = int(mlt_start2 * w / 24)
    col_end2 = int(mlt_end2 * w / 24)
    
    x_start = col_start
    x_end = col_end
    
    x_start2 = col_start2
    x_end2 = col_end2
    
    mlat_min, mlat_max = mlat_range
    
    # 计算行索引范围
    row_min = int((90 - mlat_max) * h / 40)  # 30
    row_max = int((90 - mlat_min) * h / 40)  # 50
    
    y_start = row_min
    y_end = row_max
    
    # 创建掩码
    mask = np.ones(image_shape)
    mask[y_start:y_end, x_start:x_end] = 0
    mask[y_start:y_end, x_start2:x_end2] = 0
    
    return mask

# ==================== 5. 可视化函数 ====================
def plot_comparison_curve(timestamps, cond_scores, nocond_scores, interp_scores, 
                         metric_name, metric_unit, save_path):
    """
    绘制对比曲线图（横坐标为时间）
    
    Args:
        timestamps: 时间戳列表
        cond_scores: 条件DDPM指标值列表
        nocond_scores: 无条件DDPM指标值列表
        interp_scores: 插值法指标值列表
        metric_name: 指标名称（如'SSIM', 'PSNR'）
        metric_unit: 指标单位（如'Score', 'dB'）
        save_path: 保存路径
    """
    # 转换时间戳为datetime对象
    time_objs = [convert_datetime64_to_datetime(ts) for ts in timestamps]
    
    # 将时间格式化为字符串，用于x轴标签
    # 我们只取10个均匀分布的日期作为x轴标签
    n_points = len(time_objs)
    n_labels = 10
    label_indices = np.linspace(0, n_points-1, n_labels, dtype=int)
    label_times = [time_objs[i] for i in label_indices]
    label_strs = [t.strftime("%m-%d\n%H:%M") for t in label_times]
    
    # 所有时间点的x轴位置（用于绘图）
    all_indices = np.arange(n_points)
    
    plt.figure(figsize=(14, 6))
    
    # 绘制三条曲线
    plt.plot(all_indices, cond_scores, 'o-', linewidth=2, markersize=6, 
             color='red', label='Conditional DDPM', alpha=0.8)
    plt.plot(all_indices, nocond_scores, 's-', linewidth=2, markersize=6, 
             color='green', label='Unconditional DDPM', alpha=0.8)
    plt.plot(all_indices, interp_scores, '^-', linewidth=2, markersize=6, 
             color='blue', label='Interpolation', alpha=0.8)
    
    # 计算平均值
    cond_mean = np.mean(cond_scores)
    nocond_mean = np.mean(nocond_scores)
    interp_mean = np.mean(interp_scores)
    
    # 添加平均值线
    plt.axhline(y=cond_mean, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=nocond_mean, color='green', linestyle='--', alpha=0.5, linewidth=1)
    plt.axhline(y=interp_mean, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # 添加平均值标注
    plt.text(n_points-0.5, cond_mean+0.02*cond_mean if cond_mean>0 else cond_mean-0.02*abs(cond_mean), 
             f'Cond: {cond_mean:.3f}', color='red', fontsize=10, ha='right')
    plt.text(n_points-0.5, nocond_mean+0.02*nocond_mean if nocond_mean>0 else nocond_mean-0.02*abs(nocond_mean), 
             f'Uncond: {nocond_mean:.3f}', color='green', fontsize=10, ha='right')
    plt.text(n_points-0.5, interp_mean+0.02*interp_mean if interp_mean>0 else interp_mean-0.02*abs(interp_mean), 
             f'Interp: {interp_mean:.3f}', color='blue', fontsize=10, ha='right')
    
    # 在右上角添加图例
    plt.legend(loc='upper right', fontsize=11)
    
    # 设置图形属性
    plt.xlabel('Time Points', fontsize=12)
    plt.ylabel(f'{metric_name} ({metric_unit})', fontsize=12)
    plt.title(f'{metric_name} Comparison: Conditional DDPM vs Unconditional DDPM vs Interpolation', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度为10个均匀分布的日期
    plt.xticks(label_indices, label_strs, rotation=45, fontsize=10)
    
    # 根据指标类型调整y轴范围
    if metric_name == 'SSIM' or metric_name == 'R²':
        plt.ylim(-0.1, 1.05)
    elif metric_name == 'PSNR':
        plt.ylim(0, max(max(cond_scores), max(nocond_scores), max(interp_scores)) * 1.1)
    elif metric_name == 'MAE':
        plt.ylim(0, max(max(cond_scores), max(nocond_scores), max(interp_scores)) * 1.1)
    elif metric_name == 'RMSE':
        plt.ylim(0, max(max(cond_scores), max(nocond_scores), max(interp_scores)) * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{metric_name}对比曲线图已保存: {save_path}")

# ==================== 6. 主函数：计算指标并可视化 ====================
def evaluate_and_visualize_metrics():
    """计算指标并生成可视化图表"""
    print("\n" + "="*60)
    print("计算修复指标并生成可视化图表")
    print("="*60)
    
    # 创建指标目录
    metrics_dir = os.path.join(results_dir, "metrics_visualization")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 存储结果
    timestamps_list = []
    
    # 条件DDPM指标
    cond_ssim_scores = []
    cond_psnr_scores = []
    cond_mae_scores = []
    cond_rmse_scores = []
    cond_r2_scores = []  # 新增：R²指标
    
    # 无条件DDPM指标
    nocond_ssim_scores = []
    nocond_psnr_scores = []
    nocond_mae_scores = []
    nocond_rmse_scores = []
    nocond_r2_scores = []  # 新增：R²指标
    
    # 插值法指标
    interp_ssim_scores = []
    interp_psnr_scores = []
    interp_mae_scores = []
    interp_rmse_scores = []
    interp_r2_scores = []  # 新增：R²指标
    
    num_total_points = len(mn_data)
    print(f"开始处理 {num_total_points} 个时间点...")
    
    for i in range(num_total_points):
        if i % 10 == 0:  # 每处理10个点打印一次进度
            print(f"  处理第 {i+1}/{num_total_points} 个时间点...")
        
        # 获取原始图像
        original_img = mn_data[i]
        
        # 获取时间戳
        timestamp = mn_timestamps[i]
        timestamps_list.append(timestamp)
        
        # 创建掩码
        mask = create_mask_in_mlt_region(
            original_img.shape, 
            mlat_range=(60, 80))
        
        # 创建破损图像
        corrupted_img = original_img.copy()
        corrupted_img[mask == 0] = 0
        
        # 获取太阳风数据
        solar_point = solar_data[selected_indices[i]]
        
        # 使用条件扩散模型修复
        ddpm_cond_restored = ddpm_repair(corrupted_img, mask, solar_point)
        
        # 使用无条件扩散模型修复
        ddpm_nocond_restored = ddpm_repair_nocond(corrupted_img, mask)
        
        # 使用插值方法修复
        interp_restored = interpolate_inpainting(corrupted_img, mask)
        
        # 计算条件DDPM指标
        cond_ssim = calculate_ssim(original_img, ddpm_cond_restored, mask)
        cond_psnr = calculate_psnr(original_img, ddpm_cond_restored, mask)
        cond_mae = calculate_mae(original_img, ddpm_cond_restored, mask)
        cond_rmse = calculate_rmse(original_img, ddpm_cond_restored, mask)
        cond_r2 = calculate_r2(original_img, ddpm_cond_restored, mask)  # 新增：R²
        
        # 计算无条件DDPM指标
        nocond_ssim = calculate_ssim(original_img, ddpm_nocond_restored, mask)
        nocond_psnr = calculate_psnr(original_img, ddpm_nocond_restored, mask)
        nocond_mae = calculate_mae(original_img, ddpm_nocond_restored, mask)
        nocond_rmse = calculate_rmse(original_img, ddpm_nocond_restored, mask)
        nocond_r2 = calculate_r2(original_img, ddpm_nocond_restored, mask)  # 新增：R²
        
        # 计算插值法指标
        interp_ssim = calculate_ssim(original_img, interp_restored, mask)
        interp_psnr = calculate_psnr(original_img, interp_restored, mask)
        interp_mae = calculate_mae(original_img, interp_restored, mask)
        interp_rmse = calculate_rmse(original_img, interp_restored, mask)
        interp_r2 = calculate_r2(original_img, interp_restored, mask)  # 新增：R²
        
        # 保存结果
        cond_ssim_scores.append(cond_ssim)
        cond_psnr_scores.append(cond_psnr)
        cond_mae_scores.append(cond_mae)
        cond_rmse_scores.append(cond_rmse)
        cond_r2_scores.append(cond_r2)  # 新增：R²
        
        nocond_ssim_scores.append(nocond_ssim)
        nocond_psnr_scores.append(nocond_psnr)
        nocond_mae_scores.append(nocond_mae)
        nocond_rmse_scores.append(nocond_rmse)
        nocond_r2_scores.append(nocond_r2)  # 新增：R²
        
        interp_ssim_scores.append(interp_ssim)
        interp_psnr_scores.append(interp_psnr)
        interp_mae_scores.append(interp_mae)
        interp_rmse_scores.append(interp_rmse)
        interp_r2_scores.append(interp_r2)  # 新增：R²
    
    # 生成可视化图表
    print("\n生成指标对比曲线图...")
    
    # 1. SSIM对比曲线图
    ssim_path = os.path.join(metrics_dir, "ssim_comparison_curve.png")
    plot_comparison_curve(timestamps_list, cond_ssim_scores, nocond_ssim_scores, 
                         interp_ssim_scores, 'SSIM', 'Score', ssim_path)
    
    # 2. PSNR对比曲线图
    psnr_path = os.path.join(metrics_dir, "psnr_comparison_curve.png")
    plot_comparison_curve(timestamps_list, cond_psnr_scores, nocond_psnr_scores, 
                         interp_psnr_scores, 'PSNR', 'dB', psnr_path)
    
    # 3. MAE对比曲线图
    mae_path = os.path.join(metrics_dir, "mae_comparison_curve.png")
    plot_comparison_curve(timestamps_list, cond_mae_scores, nocond_mae_scores, 
                         interp_mae_scores, 'MAE', 'Value', mae_path)
    
    # 4. RMSE对比曲线图
    rmse_path = os.path.join(metrics_dir, "rmse_comparison_curve.png")
    plot_comparison_curve(timestamps_list, cond_rmse_scores, nocond_rmse_scores, 
                         interp_rmse_scores, 'RMSE', 'Value', rmse_path)
    
    # 5. R²对比曲线图（新增）
    r2_path = os.path.join(metrics_dir, "r2_comparison_curve.png")
    plot_comparison_curve(timestamps_list, cond_r2_scores, nocond_r2_scores, 
                         interp_r2_scores, 'R²', 'Score', r2_path)
    
    # 打印R²的统计摘要
    print("\n" + "="*60)
    print("R²统计摘要:")
    print("="*60)
    print(f"条件DDPM R²范围: [{min(cond_r2_scores):.3f}, {max(cond_r2_scores):.3f}], 平均值: {np.mean(cond_r2_scores):.3f}")
    print(f"无条件DDPM R²范围: [{min(nocond_r2_scores):.3f}, {max(nocond_r2_scores):.3f}], 平均值: {np.mean(nocond_r2_scores):.3f}")
    print(f"插值法 R²范围: [{min(interp_r2_scores):.3f}, {max(interp_r2_scores):.3f}], 平均值: {np.mean(interp_r2_scores):.3f}")
    
    print("\n所有对比曲线图已保存到:", metrics_dir)
    print("评估完成!")

# ==================== 7. 主程序入口 ====================
if __name__ == "__main__":
    evaluate_and_visualize_metrics()