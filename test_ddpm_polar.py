import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as opt
import torch_npu
from torch.utils.data import DataLoader, Dataset
from models.simplenet import UNet as UNet_nocond
from models.ddpm_nocond import DDPM_nocond
from models.unet_v2 import UNet
from models.ddpm import DDPM
import os
from datetime import datetime
import pandas as pd
from utils.normlize import normalize, denormalize
from data.dataset_diff2 import load_all_aurora_data, OmniDataset
warnings.filterwarnings('ignore')

# ==================== 1. 参数设置 ====================
num_test_images = 5  # 测试5张图像
results_dir = "/home/docker/code/Aurora_DDPM_final/res/polar/repaired_ckptv0_unetv2/"
os.makedirs(results_dir, exist_ok=True)
num_steps = 300
repaint_steps = 10
jump_len = 10
N = 10
n_samples = 1
device = "npu:0"

# ====================  条件扩散 ====================
model_save_path = "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv0_unetv2/aurora_diff_best.pth"
unet = UNet(1, 1)
ddpm = DDPM(unet, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path, map_location=device)
ddpm.load_state_dict(checkpoint['model_state_dict'],strict=False)
ddpm.eval()
ddpm.to(device)

# ====================  无条件扩散 ====================
model_save_path_nocond = "/home/docker/code/Aurora_DDPM_final/ckpt/uncond/ckptv3_simple/aurora_diff_best.pth"
unet_nocond = UNet_nocond(1, 1)
ddpm_nocond = DDPM_nocond(unet_nocond, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path_nocond, map_location=device)
ddpm_nocond.load_state_dict(checkpoint['model_state_dict'], strict=False)
ddpm_nocond.eval()
ddpm_nocond.to(device)

# ==================== 加载测试数据 ====================
print("加载测试数据...")
test_data_all = np.load(
    "/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/resampled_5min_1996_0405.npy",
    allow_pickle=True
)
data_mn_all,_ = load_all_aurora_data(years=[1996], months=range(1, 3))

mn_mean = data_mn_all.mean()
mn_std = data_mn_all.std()
mn_var = data_mn_all.var()
mn_max = data_mn_all.max()
mn_min = data_mn_all.min()

# 从数据中提取前5张图像
polar_timestamps = test_data_all['utc']
polar_data_all = np.stack(test_data_all['aurora_image'], axis=0).astype(np.float32)

polar_mean = polar_data_all.mean()
polar_std = polar_data_all.std()
polar_var = polar_data_all.var()
polar_max = polar_data_all.max()
polar_min = polar_data_all.min()

#data_mn_all = (data_mn_all - mn_min) / (mn_max - mn_min) * (polar_max - polar_min) + polar_min
#polar_data_transfer = ((polar_data_all - polar_mean) /polar_std) * mn_std + mn_mean

# data1_path = "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/aurora_img_20050101.npy"
# mn_datas = np.load(data1_path)
# omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_1min_pro/2005/omni_20050101_1min.npy"
# omni_data = np.load(omni_path)
# mn_time = omni_data['utc']

solar_fields = ['Bx', 'By', 'Bz', 'V', 'P']
solar_components = []
for field in solar_fields:
    if field in test_data_all.dtype.names:
        field_data = test_data_all[field]
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

# 每隔20张取一张，确保不超过数据范围
interval = 20
selected_indices = []
current_idx = 0

while len(selected_indices) < num_test_images :
    selected_indices.append(current_idx)
    current_idx += interval


# 选择数据
mn_timestamps = polar_timestamps[selected_indices]
mn_data = polar_data_all[selected_indices]
solar_data = solar_data[selected_indices]

normalizer_mn = normalize(data_mn_all)
denormalizer_mn = denormalize(data_mn_all)
print(f"加载了 {mn_data.shape[0]} 张测试图像")
print(f"图像形状: {mn_data.shape}")

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
    
    mlt_start2 = 1
    mlt_end2 = 5
    
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
    #mask[y_start:y_end, x_start:x_end] = 0
    mask[y_start:y_end, x_start2:x_end2] = 0
    
    return mask

# ==================== 3. 定义评估指标函数 ====================
def calculate_psnr(img1, img2, mask):
    """计算PSNR (峰值信噪比)"""
    mse = np.mean(((img1 - img2) ** 2)[mask == 0])
    if mse == 0:
        return float('inf')
    max_pixel = np.max([np.max(img1), np.max(img2)])
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, mask):
    """计算SSIM (结构相似性)"""
    # 简化的SSIM计算，只计算缺失区域
    img1_masked = img1[mask == 0].flatten()
    img2_masked = img2[mask == 0].flatten()
    
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
    C1 = (0.01 * np.max([np.max(img1), np.max(img2)])) ** 2
    C2 = (0.03 * np.max([np.max(img1), np.max(img2)])) ** 2
    
    # 计算SSIM
    ssim_numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return ssim_numerator / ssim_denominator

def calculate_lpips_simple(img1, img2, mask):
    """
    简化的LPIPS计算 (感知相似度)
    由于我们没有预训练模型，这里使用梯度信息的相似度作为替代
    """
    from scipy.ndimage import sobel
    
    # 只计算缺失区域
    img1_masked = img1.copy()
    img2_masked = img2.copy()
    img1_masked[mask == 1] = 0
    img2_masked[mask == 1] = 0
    
    # 计算梯度
    grad1_x = sobel(img1_masked, axis=1)
    grad1_y = sobel(img1_masked, axis=0)
    grad2_x = sobel(img2_masked, axis=1)
    grad2_y = sobel(img2_masked, axis=0)
    
    # 计算梯度差异
    grad_diff = np.sqrt((grad1_x - grad2_x)**2 + (grad1_y - grad2_y)**2)
    
    # 只计算缺失区域的梯度差异
    grad_diff_masked = grad_diff[mask == 0]
    if len(grad_diff_masked) == 0:
        return 0
    
    lpips_score = np.mean(grad_diff_masked)
    # 归一化到[0,1]范围
    max_grad = np.max([np.max(np.abs(grad1_x)), np.max(np.abs(grad1_y)), 
                      np.max(np.abs(grad2_x)), np.max(np.abs(grad2_y))])
    if max_grad > 0:
        lpips_score = lpips_score / max_grad
    
    return lpips_score

def calculate_metrics(original, restored, mask, method_name):
    """计算所有指标"""
    metrics = {
        'method': method_name,
        'psnr': calculate_psnr(original, restored, mask),
        'ssim': calculate_ssim(original, restored, mask),
        'lpips': calculate_lpips_simple(original, restored, mask)
    }
    return metrics

# ==================== 4. 定义插值修复函数 ====================
def interpolate_inpainting(corrupted_img, mask):
    """
    使用双线性插值进行修复
    """
    # 获取已知点和未知点的坐标
    known_coords = np.column_stack(np.where(mask == 1))
    unknown_coords = np.column_stack(np.where(mask == 0))
    
    if len(known_coords) == 0 or len(unknown_coords) == 0:
        return corrupted_img.copy()
    
    # 获取已知点的值
    known_values = corrupted_img[mask == 1]
    
    # 使用griddata进行插值
    interpolated_values = griddata(
        known_coords, 
        known_values, 
        unknown_coords, 
        method='linear',
        fill_value=0  # 对于边缘点，用0填充
    )
    
    # 创建修复后的图像
    restored_img = corrupted_img.copy()
    restored_img[mask == 0] = interpolated_values
    
    return restored_img

def convert_datetime64_to_datetime(dt64):
    """将 numpy.datetime64 转换为 datetime.datetime"""
    if dt64 is None:
        return None
    if isinstance(dt64, datetime):
        return dt64
    import pandas as pd
    return pd.Timestamp(dt64).to_pydatetime()

# ==================== 5. 修改画图函数 ====================
def plot_aurora_comparison(timestamp, original_img, corrupted_img, restored_img, mask, save_path):
    """
    绘制对比图：原图、破损图、修复图
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': 'polar'})
    timestamp = convert_datetime64_to_datetime(timestamp)
    # 自定义极光颜色映射
    colors = [
        (0, 0, 0), (0, 0, 0.3), (0, 0, 0.8), (0, 0.5, 1),
        (0, 1, 1), (0.5, 1, 0.5), (1, 1, 0),
        (1, 0.5, 0), (1, 0, 0), (0.8, 0.8, 0.8)
    ]
    custom_cmap = LinearSegmentedColormap.from_list('aurora_cmap', colors, N=256)
    
    # 极坐标网格
    mlat = np.linspace(50, 90, 80)
    mlt = np.linspace(0, 24, 96)
    MLAT, MLT = np.meshgrid(mlat, mlt)
    theta = (MLT / 24.0) * 2 * np.pi - np.pi/2
    r = (90 - MLAT) / 40.0
    
    # 确定统一的颜色范围
    vmin = 0
    vmax = 5
    
    # 1. 绘制原图
    ax1 = axes[0]
    im1 = ax1.pcolormesh(theta, r, original_img.T, cmap=custom_cmap, 
                        shading='auto', vmin=vmin, vmax=vmax)
    ax1.set_theta_zero_location('S')
    ax1.set_theta_direction(1)
    ax1.set_ylim(0, 1)
    ax1.set_yticklabels([])
    # 设置角度刻度为地方时
    hour_ticks = np.arange(0, 24, 1)
    angle_ticks = (hour_ticks / 24.0) * 360  # 转换为角度
    ax1.set_xticks(np.deg2rad(angle_ticks))

    # 设置刻度标签 - 只显示0,6,12,18，其他为空
    mlt_labels = []
    for hour in hour_ticks:
        if hour in [0, 6, 12, 18]:
            mlt_labels.append(str(hour))
        else:
            mlt_labels.append('')
    ax1.set_xticklabels(mlt_labels, fontsize=10)

    # 7. 设置8个径向网格线（纬度圈）的位置
    # 从50°到85°，每5°一个，共8个圈
    lat_circles = [50, 55, 60, 65, 70, 75, 80, 85]
    radial_ticks = [(90 - lat) / 40.0 for lat in lat_circles]  # 转换为半径

    # 设置径向网格线的位置和标签
    ax1.set_rticks(radial_ticks)
    ax1.set_yticklabels([f'{lat}°' for lat in lat_circles], 
                    fontsize=5, color='white')
    ax1.set_title('Original Aurora', fontsize=14, fontweight='bold')
    
    # 2. 绘制破损图（用红框标出缺失区域）
    ax2 = axes[1]
    im2 = ax2.pcolormesh(theta, r, corrupted_img.T, cmap=custom_cmap, 
                        shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_theta_zero_location('S')
    ax2.set_theta_direction(1)
    ax2.set_ylim(0, 1)
    ax2.set_yticklabels([])
    # 设置角度刻度为地方时
    ax2.set_xticks(np.deg2rad(angle_ticks))
    ax2.set_xticklabels(mlt_labels, fontsize=10)
    # 设置径向网格线的位置和标签
    ax2.set_rticks(radial_ticks)
    ax2.set_yticklabels([f'{lat}°' for lat in lat_circles], 
                    fontsize=5, color='white')
    ax2.set_title('Corrupted Aurora (Masked Region in Red)', fontsize=14, fontweight='bold')
    
    # 在破损图上绘制红框标记缺失区域
    # 将掩码坐标转换为极坐标
    mask_y, mask_x = np.where(mask == 0)
    if len(mask_y) > 0 and len(mask_x) > 0:
        # 计算缺失区域的边界
        y_min, y_max = mask_y.min(), mask_y.max()
        x_min, x_max = mask_x.min(), mask_x.max()
        
        # 获取边界点的极坐标
        # 注意：这里需要处理边界，避免索引超出范围
        x_min = max(0, x_min)
        x_max = min(theta.shape[0]-1, x_max)
        y_min = max(0, y_min)
        y_max = min(theta.shape[1]-1, y_max)
        
        # 方法1：绘制精确的边界（使用网格边界）
        # 创建边界点：内边界和外边界都要考虑
        n_points = 50  # 每边采样的点数
        
        # 上边界（纬度较小，r较大）
        theta_top = theta[x_min:x_max+1, y_min]
        r_top = r[x_min:x_max+1, y_min]
        
        # 下边界（纬度较大，r较小）
        theta_bottom = theta[x_min:x_max+1, y_max]
        r_bottom = r[x_min:x_max+1, y_max]
        
        # 左边界（经度方向）
        theta_left = theta[x_min, y_min:y_max+1]
        r_left = r[x_min, y_min:y_max+1]
        
        # 右边界（经度方向）
        theta_right = theta[x_max, y_min:y_max+1]
        r_right = r[x_max, y_min:y_max+1]
        
        # 连接所有边界点形成闭合曲线
        # 顺序：上边界 -> 右边界 -> 下边界（反向） -> 左边界（反向）
        theta_boundary = np.concatenate([
            theta_top,  # 上边界
            theta_right[1:],  # 右边界（跳过第一个点，因为已经是右上角）
            np.flip(theta_bottom)[1:],  # 下边界反向（跳过第一个点）
            np.flip(theta_left)[1:-1]  # 左边界反向（跳过第一个和最后一个点）
        ])
        
        r_boundary = np.concatenate([
            r_top,  # 上边界
            r_right[1:],  # 右边界
            np.flip(r_bottom)[1:],  # 下边界反向
            np.flip(r_left)[1:-1]  # 左边界反向
        ])
        
        # 绘制红色边界框
        ax2.plot(theta_boundary, r_boundary, 'r-', linewidth=2, alpha=0.8)
    
    # 3. 绘制修复图
    ax3 = axes[2]
    im3 = ax3.pcolormesh(theta, r, restored_img.T, cmap=custom_cmap, 
                        shading='auto', vmin=vmin, vmax=vmax)
    ax3.set_theta_zero_location('S')
    ax3.set_theta_direction(1)
    ax3.set_ylim(0, 1)
    ax3.set_yticklabels([])
    # 设置角度刻度为地方时
    ax3.set_xticks(np.deg2rad(angle_ticks))
    ax3.set_xticklabels(mlt_labels, fontsize=10)
    # 设置径向网格线的位置和标签
    ax3.set_rticks(radial_ticks)
    ax3.set_yticklabels([f'{lat}°' for lat in lat_circles], 
                    fontsize=5, color='white')
    ax3.set_title('Restored Aurora', fontsize=14, fontweight='bold')
    
    # 添加时间戳
    from datetime import datetime
    if isinstance(timestamp, np.datetime64):
        timestamp = timestamp.astype(datetime)
    time_str = timestamp.strftime("%Y-%m-%d %H:%M UT")
    fig.suptitle(f'Aurora Image Restoration - {time_str}', fontsize=16, fontweight='bold', y=1.02)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.98, 0.25, 0.03, 0.5]) # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax, label='Energy Flux (ergs cm⁻² s⁻¹)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存对比图: {save_path}")

# ==================== 新增：绘制DDPM修复散点图函数 ====================
def plot_ddpm_scatter(timestamp, original_img, ddpm_restored, mask, save_path):
    """
    绘制DDPM修复的预测值-真实值散点图（仅限挖去部分）
    
    Args:
        timestamp: 时间戳
        original_img: 原始图像
        ddpm_restored: DDPM修复的图像
        mask: 掩码（0表示挖去部分）
        save_path: 保存路径
    """
    # 提取挖去部分的真实值和预测值
    true_values = original_img[mask == 0].flatten()
    pred_values = ddpm_restored[mask == 0].flatten()
    
    if len(true_values) == 0:
        print(f"警告：图像 {save_path} 没有挖去部分的数据")
        return
    
    # 计算R²值
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        r2 = 1 - (ss_res / ss_tot)
        return max(r2, -1.0)
    
    r2_score = calculate_r2(true_values, pred_values)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点图
    scatter = ax.scatter(true_values, pred_values, alpha=0.5, s=20, 
                         c='blue', edgecolors='none')
    
    # 绘制y=x参考线（完美预测线）
    min_val = min(np.min(true_values), np.min(pred_values))
    max_val = max(np.max(true_values), np.max(pred_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # 添加时间戳和R²值
    timestamp = convert_datetime64_to_datetime(timestamp)
    if isinstance(timestamp, datetime):
        time_str = timestamp.strftime("%Y-%m-%d %H:%M UT")
    else:
        time_str = str(timestamp)
    
    # 在左上角添加R²值
    ax.text(0.05, 0.95, f'R² = {r2_score:.4f}', transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置图形属性
    ax.set_xlabel('True Values (Original)', fontsize=14)
    ax.set_ylabel('Predicted Values (DDPM Restored)', fontsize=14)
    ax.set_title(f'DDPM Restoration: Predicted vs True Values\n{time_str}', 
                 fontsize=16, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # 设置相同的x和y轴范围
    margin = 0.05 * (max_val - min_val)
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存DDPM散点图: {save_path}")
    
    return r2_score

# ====================  定义ddpm修复函数 ====================
def ddpm_repaired(input_data, mask,solar_point):
    #input_data = ((input_data - polar_mean) /polar_std) * mn_std + mn_mean
    input_data = (input_data - polar_min) / (polar_max - polar_min) * (mn_max - mn_min) + mn_min
    input_data = normalizer_mn(input_data)
    input_data = np.expand_dims(input_data, axis=(0,1))
    mask = np.expand_dims(mask, axis=(0, 1))
    input_data = torch.tensor(input_data).float().to(device)
    mask = torch.tensor(mask).float().to(device)
    solar_point = solar_point.unsqueeze(0).to(device)
    with torch.no_grad():
        # 使用ddpm.sample进行修补
        repaired = ddpm.sample(
            input_data,
            mask,
            solar_point,
            num_inference_steps=num_steps,
            n_sample=n_samples,
            j=jump_len,
            r=repaint_steps,
        )
    repaired_np = repaired.cpu().numpy().squeeze()
    repaired_flux = denormalizer_mn(repaired_np)
    #repaired_flux = (repaired_flux - mn_mean) / mn_std * polar_std + polar_mean
    repaired_flux = (repaired_flux - mn_min) / (mn_max - mn_min) * (polar_max - polar_min) + polar_min
    return repaired_flux

def ddpm_repair_nocond(input_data, mask):
    """扩散模型修复函数（无条件）"""
    #input_data = ((input_data - polar_mean) /polar_std) * mn_std + mn_mean
    input_data = (input_data - polar_min) / (polar_max - polar_min) * (mn_max - mn_min) + mn_min
    input_data_norm = normalizer_mn(input_data)
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
    repaired_flux = denormalizer_mn(repaired_np)
    # repaired_flux = (repaired_flux - mn_mean) / mn_std * polar_std + polar_mean
    repaired_flux = (repaired_flux - mn_min) / (mn_max - mn_min) * (polar_max - polar_min) + polar_min
    return repaired_flux

# ==================== 6. 评估函数 ====================
def evaluate_inpainting():
    """
    主评估函数：对5张图像进行评估
    """
    print("\n" + "="*60)
    print("开始极光图像修复评估")
    print("="*60)
    
    all_metrics = []
    all_r2_scores = []  # 存储所有图像的R²分数
    
    for i in range(num_test_images):
        print(f"\n评估第 {i+1}/{num_test_images} 张图像...")
        
        # 获取原始图像
        original_img = mn_data[i]
        
        mask = create_mask_in_mlt_region(
            original_img.shape, 
            mlat_range=(68, 75)) 
        
        # 创建破损图像（缺失部分用0填充）
        corrupted_img = original_img.copy()
        corrupted_img[mask == 0] = 0
        
        # 使用扩散模型修复
        print("  使用条件扩散模型修复中...")
        solar_point = solar_data[i]
        ddpm_restored = ddpm_repaired(corrupted_img, mask,solar_point)
        # ddpm_restored = (ddpm_restored - mn_mean) / mn_std * polar_std + polar_mean
        # ddpm_restored[mask == 0] = ddpm_restored[mask == 0]
        print("  使用无条件扩散模型修复中...")
        ddpm_restored_nocond = ddpm_repair_nocond(corrupted_img, mask)
        
        # 使用插值方法修复
        print("  使用插值方法修复中...")
        interpolated_restored = interpolate_inpainting(corrupted_img, mask)
        
        # 计算指标
        ddpm_metrics = calculate_metrics(original_img, ddpm_restored, mask, "DDPM")
        ddpm_nocond_metrics = calculate_metrics(original_img, ddpm_restored_nocond, mask, "DDPM_NoCond")
        interp_metrics = calculate_metrics(original_img, interpolated_restored, mask, "Interpolation")
        
        # 保存到列表
        all_metrics.append({
            'image_id': i,
            'timestamp': mn_timestamps[i],
            'ddpm': ddpm_metrics,
            'ddpm_nocond': ddpm_nocond_metrics,
            'interpolation': interp_metrics
        })
        
        # 打印当前图像的指标
        print(f"  图像 {i+1} 指标:")
        print(f"    条件DDPM模型 - PSNR: {ddpm_metrics['psnr']:.2f} dB, "
              f"SSIM: {ddpm_metrics['ssim']:.4f}, "
              f"LPIPS: {ddpm_metrics['lpips']:.4f}")
        print(f"    无条件DDPM模型 - PSNR: {ddpm_nocond_metrics['psnr']:.2f} dB, "
              f"SSIM: {ddpm_nocond_metrics['ssim']:.4f}, "
              f"LPIPS: {ddpm_nocond_metrics['lpips']:.4f}")
        print(f"    插值方法 - PSNR: {interp_metrics['psnr']:.2f} dB, "
              f"SSIM: {interp_metrics['ssim']:.4f}, "
              f"LPIPS: {interp_metrics['lpips']:.4f}")
        
        # 绘制并保存对比图
        save_path = os.path.join(results_dir, f"aurora_comparison_{i+1}.png")
        plot_aurora_comparison(
            mn_timestamps[i],
            original_img,
            corrupted_img,
            ddpm_restored,
            mask,
            save_path
        )
        
        # 新增：绘制DDPM修复散点图
        scatter_save_path = os.path.join(results_dir, f"ddpm_scatter_{i+1}.png")
        r2_score = plot_ddpm_scatter(
            mn_timestamps[i],
            original_img,
            ddpm_restored,
            mask,
            scatter_save_path
        )
        all_r2_scores.append(r2_score)
        print(f"    DDPM修复R²分数: {r2_score:.4f}")
        
        scatter_save_path = os.path.join(results_dir, f"ddpm_nocond_{i+1}.png")
        r2_score = plot_ddpm_scatter(
            mn_timestamps[i],
            original_img,
            ddpm_restored_nocond,
            mask,
            scatter_save_path
        )
        all_r2_scores.append(r2_score)
        print(f"    DDPM无条件修复R²分数: {r2_score:.4f}")
    
    # ==================== 7. 打印总体统计结果 ====================
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    # 计算平均指标
    ddpm_psnr_avg = np.mean([m['ddpm']['psnr'] for m in all_metrics])
    ddpm_ssim_avg = np.mean([m['ddpm']['ssim'] for m in all_metrics])
    ddpm_lpips_avg = np.mean([m['ddpm']['lpips'] for m in all_metrics])
    
    ddpm_nocond_psnr_avg = np.mean([m['ddpm_nocond']['psnr'] for m in all_metrics])
    ddpm_nocond_ssim_avg = np.mean([m['ddpm_nocond']['ssim'] for m in all_metrics])
    ddpm_nocond_lpips_avg = np.mean([m['ddpm_nocond']['lpips'] for m in all_metrics])
    
    interp_psnr_avg = np.mean([m['interpolation']['psnr'] for m in all_metrics])
    interp_ssim_avg = np.mean([m['interpolation']['ssim'] for m in all_metrics])
    interp_lpips_avg = np.mean([m['interpolation']['lpips'] for m in all_metrics])
    
    # 计算R²平均分数
    if all_r2_scores:
        r2_avg = np.mean(all_r2_scores)
        r2_min = np.min(all_r2_scores)
        r2_max = np.max(all_r2_scores)
    
    print(f"\nDDPM模型平均指标:")
    print(f"  PSNR: {ddpm_psnr_avg:.2f} dB")
    print(f"  SSIM: {ddpm_ssim_avg:.4f}")
    print(f"  LPIPS: {ddpm_lpips_avg:.4f}")
    if all_r2_scores:
        print(f"  R²: {r2_avg:.4f} (范围: {r2_min:.4f} - {r2_max:.4f})")
    
    
    print(f"\n无条件DDPM模型平均指标:")
    print(f"  PSNR: {ddpm_nocond_psnr_avg:.2f} dB")
    print(f"  SSIM: {ddpm_nocond_ssim_avg:.4f}")
    print(f"  LPIPS: {ddpm_nocond_lpips_avg:.4f}")
    
    print(f"\n插值方法平均指标:")
    print(f"  PSNR: {interp_psnr_avg:.2f} dB")
    print(f"  SSIM: {interp_ssim_avg:.4f}")
    print(f"  LPIPS: {interp_lpips_avg:.4f}")
    
    # 计算提升百分比
    psnr_improvement = ((ddpm_psnr_avg - interp_psnr_avg) / interp_psnr_avg) * 100
    ssim_improvement = ((ddpm_ssim_avg - interp_ssim_avg) / interp_ssim_avg) * 100
    lpips_improvement = ((interp_lpips_avg - ddpm_lpips_avg) / interp_lpips_avg) * 100  # LPIPS越低越好
    
    print(f"\nDDPM相比插值方法的提升:")
    print(f"  PSNR: {psnr_improvement:+.2f}%")
    print(f"  SSIM: {ssim_improvement:+.2f}%")
    print(f"  LPIPS: {lpips_improvement:+.2f}% (降低)")
    
    # 保存详细结果到文件
    results_file = os.path.join(results_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("极光图像修复评估结果\n")
        f.write("="*50 + "\n\n")
        
        for idx, m in enumerate(all_metrics):
            f.write(f"图像 {m['image_id']+1} - {m['timestamp']}\n")
            f.write(f"  DDPM模型: PSNR={m['ddpm']['psnr']:.2f} dB, "
                   f"SSIM={m['ddpm']['ssim']:.4f}, LPIPS={m['ddpm']['lpips']:.4f}")
            if idx < len(all_r2_scores):
                f.write(f", R²={all_r2_scores[idx]:.4f}\n")
            else:
                f.write("\n")
            f.write(f"  插值方法: PSNR={m['interpolation']['psnr']:.2f} dB, "
                   f"SSIM={m['interpolation']['ssim']:.4f}, LPIPS={m['interpolation']['lpips']:.4f}\n\n")
        
        f.write("\n平均指标:\n")
        f.write(f"DDPM模型: PSNR={ddpm_psnr_avg:.2f} dB, SSIM={ddpm_ssim_avg:.4f}, LPIPS={ddpm_lpips_avg:.4f}")
        if all_r2_scores:
            f.write(f", R²={r2_avg:.4f} (范围: {r2_min:.4f} - {r2_max:.4f})\n")
        else:
            f.write("\n")
        f.write(f"插值方法: PSNR={interp_psnr_avg:.2f} dB, SSIM={interp_ssim_avg:.4f}, LPIPS={interp_lpips_avg:.4f}\n\n")
        
        f.write(f"DDPM相比插值方法的提升:\n")
        f.write(f"  PSNR: {psnr_improvement:+.2f}%\n")
        f.write(f"  SSIM: {ssim_improvement:+.2f}%\n")
        f.write(f"  LPIPS: {lpips_improvement:+.2f}% (降低)\n")
    
    print(f"\n详细结果已保存到: {results_file}")
    print(f"对比图已保存到: {results_dir}/")
    print(f"DDPM散点图已保存到: {results_dir}/")
    print("\n评估完成!")

# ==================== 8. 主程序入口 ====================
if __name__ == "__main__":
    # 运行评估
    evaluate_inpainting()