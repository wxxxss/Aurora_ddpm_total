import os
import torch
import numpy as np
from torch.utils.data import  Dataset,DataLoader

class TrainDataset(Dataset):
    def __init__(self, datas, split='train',train_ratio=0.8):
        self.datas = datas
        self.split = split
        self.train_ratio = train_ratio
        self.num_masks_per_sample = 20
        log_data = np.log1p(self.datas)
        self.log_min = float(log_data.min())
        self.log_max = float(log_data.max())
        self.log_range = max(self.log_max - self.log_min, 1e-6)
        self.precomputed_masks = self._precompute_masks()
        
        n_total = len(datas)
        n_train = int(n_total * train_ratio)

        if split == 'train':
            self.indices = datas[:n_train]
        elif split == 'val':
            self.indices = datas[n_train:]
        
        
    def _transform_aurora(self, x: np.ndarray) -> np.ndarray:
        x = np.log1p(x)
        x = (x - self.log_min) / self.log_range
        x = np.clip(x, 0.0, 1.0)
        return x.astype(np.float32)
    
    def generate_aurora_specific_mask(self, data_tensor):
        """专门针对极光数据的mask生成"""
        H, W = data_tensor.shape[1], data_tensor.shape[2]
        mask = np.ones((1, H, W), dtype=np.float32)
        
        # 检测极光区域
        # 使用更敏感的阈值检测极光
        aurora_threshold = 0.01  # 更低的阈值，检测更弱的极光
        aurora_mask = (data_tensor[0] > aurora_threshold).numpy()
        aurora_ratio = aurora_mask.mean()
        #print(f"极光覆盖比例: {aurora_ratio:.2%}")
        # 统计信息
        stats = {
            'aurora_ratio': aurora_ratio,
            'total_masked': 0,
            'aurora_masked': 0,
            'background_masked': 0
        }
        
        # 如果几乎没有极光，直接在随机位置挖大洞
        if aurora_ratio < 0.05:
            # 挖3-5个大洞，每个覆盖10%-20%的区域
            num_holes = np.random.randint(3, 6)
            for _ in range(num_holes):
                h_size = int(H * np.random.uniform(0.1, 0.2))
                w_size = int(W * np.random.uniform(0.1, 0.2))
                
                h_start = np.random.randint(0, H - h_size)
                w_start = np.random.randint(0, W - w_size)
                
                mask[:, h_start:h_start+h_size, w_start:w_start+w_size] = 0
                stats['total_masked'] += h_size * w_size
        
        else:
            # 1. 在极光区域挖大洞（重点）
            aurora_indices = np.argwhere(aurora_mask)
            
            # 挖掉60%-80%的极光区域
            aurora_mask_ratio = np.random.uniform(0.6, 0.8)
            num_aurora_to_mask = int(len(aurora_indices) * aurora_mask_ratio)
            
            # 分批挖洞，每批挖一个区域
            batch_size = max(1, num_aurora_to_mask // 20)  # 分成大约20批
            
            for i in range(0, num_aurora_to_mask, batch_size):
                if i >= len(aurora_indices):
                    break
                    
                # 选择一个中心点
                center_idx = np.random.randint(0, len(aurora_indices))
                h_center, w_center = aurora_indices[center_idx]
                
                # 挖一个中等大小的洞
                h_size = np.random.randint(10, 20)  # 5-15像素
                w_size = np.random.randint(10, 20)  # 5-15像素
                
                h_start = max(0, h_center - h_size//2)
                h_end = min(H, h_start + h_size)
                w_start = max(0, w_center - w_size//2)
                w_end = min(W, w_start + w_size)
                
                mask[:, h_start:h_end, w_start:w_start] = 0
                stats['aurora_masked'] += (h_end - h_start) * (w_end - w_start)
            
            # 2. 在背景区域挖少量洞（20%-40%）
            background_mask = ~aurora_mask
            background_indices = np.argwhere(background_mask)
            
            if len(background_indices) > 0:
                background_mask_ratio = np.random.uniform(0.2, 0.4)
                num_background_to_mask = int(len(background_indices) * background_mask_ratio)
                
                # 随机选择背景像素挖洞
                selected_indices = np.random.choice(
                    len(background_indices),
                    size=min(num_background_to_mask, len(background_indices)),
                    replace=False
                )
                
                for idx in selected_indices:
                    h, w = background_indices[idx]
                    
                    # 背景区域挖小洞
                    h_size = np.random.randint(2, 5)
                    w_size = np.random.randint(2, 5)
                    
                    h_start = max(0, h - h_size//2)
                    h_end = min(H, h_start + h_size)
                    w_start = max(0, w - w_size//2)
                    w_end = min(W, w_start + w_size)
                    
                    mask[:, h_start:h_end, w_start:w_start] = 0
                    stats['background_masked'] += (h_end - h_start) * (w_end - w_start)
        
        # 确保总mask比例至少70%
        current_mask_ratio = 1 - mask.mean()
        if current_mask_ratio < 0.7:
            # 添加更多随机mask
            needed_ratio = 0.7 - current_mask_ratio
            pixels_to_add = int(needed_ratio * H * W)
            
            # 使用numpy向量化操作添加随机像素
            random_mask = np.random.random((H, W)) < (pixels_to_add / (H * W))
            mask[0] = np.minimum(mask[0], 1 - random_mask.astype(np.float32))
        
        # 添加随机噪声点（模拟传感器噪声）
        if np.random.random() < 0.5:
            noise_mask = np.random.random((H, W)) < 0.01  # 1%的随机噪声点
            mask[0] = np.minimum(mask[0], 1 - noise_mask.astype(np.float32))
        
        # 计算最终统计
        stats['total_masked_ratio'] = 1 - mask.mean()
        stats['aurora_masked_ratio'] = stats['aurora_masked'] / (H * W) if aurora_ratio > 0 else 0
        stats['background_masked_ratio'] = stats['background_masked'] / (H * W)
        
        # 打印统计信息（用于调试）
        if np.random.random() < 0.01:  # 1%的概率打印，避免输出太多
            print(f"Mask统计: 总mask比例={stats['total_masked_ratio']:.1%}, "
                f"极光区域mask比例={stats['aurora_masked_ratio']:.1%}")
        
        return mask
    
    def _precompute_masks(self):
        """为每个样本预先生成多个mask"""
        print("预先生成mask...")
        all_masks = []
        
        for idx in range(len(self.datas)):
            if idx % 100 == 0:
                print(f"处理样本 {idx}/{len(self.datas)}")
            
            # 转换数据
            #data = self._transform_aurora(self.datas[idx])
            data = torch.from_numpy(self.datas[idx]).float()
            # 为这个样本生成多个mask
            sample_masks = []
            for _ in range(self.num_masks_per_sample):
                mask = self.generate_aurora_specific_mask(data.reshape(1, data.shape[0], data.shape[1]))
                sample_masks.append(mask)
            
            all_masks.append(sample_masks)
        
        print("mask生成完成")
        return all_masks
    
    def __len__(self):
        # 注意：这里我们返回原始数据长度，不是乘以mask数量
        return self.datas.shape[0]

    def __getitem__(self, idx):
        data = self.datas[idx]
        data = self._transform_aurora(data)
        data = torch.from_numpy(data).float()
        # 添加通道维度
        data = data.unsqueeze(0)  # 形状: (1, H, W)
        
        # 随机选择一个预先生成的mask
        mask_idx = np.random.randint(0, self.num_masks_per_sample)
        mask = torch.from_numpy(self.precomputed_masks[idx][mask_idx]).float()
        
        return data, mask
    
class OmniDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        self.min = datas.min(axis=0)
        self.max = datas.max(axis=0)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1.0
        
    def _transform_aurora(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.min) / self.range
        x = np.clip(x, 0.0, 1.0)
        # return (x * 2.0 - 1.0).astype(np.float32)
        return x.astype(np.float32)
    
    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        data = self.datas[idx]
        data = self._transform_aurora(data)
        data = torch.from_numpy(data).float()
        return data


    
# 加载所有数据
def load_all_aurora_data(years=[1996, 1997, 1998]):
    """加载多个月份的数据，并间隔采样（每小时一个点）"""
    solar_fields = ['Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'P']
    all_data = []
    omni_data = []
    base_path = "/home/docker/data/private/AuroraData/generated_aurora_data"
    omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_5min"
    for year in years:
        year_path = os.path.join(base_path, f"{year}_omni_aurora")
        omni_year_path = os.path.join(omni_path, f"{year}")
        if not os.path.exists(year_path):
            continue
            
        for month in range(1, 3):
            filename = f"aurora_img_{year}{month:02d}01.npy"
            filepath = os.path.join(year_path, filename)
            
            omni_filename = f"omni_{year}{month:02d}01_5min.npy"
            omni_filepath = os.path.join(omni_year_path, omni_filename)
            try:
                data = np.load(filepath)
                # 间隔12个点采样（每小时一个点，假设原始是5分钟数据）
                data_sampled = data[::12]
                all_data.append(data_sampled)
                print(f"加载: {filename}, 原始形状: {data.shape}, 采样后: {data_sampled.shape}")
            except Exception as e:
                print(f"加载失败 {filename}: {e}")
            try:
                data2 = np.load(omni_filepath, allow_pickle=True)
                solar_components = []
                for field in solar_fields:
                    if field in data2.dtype.names:
                        field_data = data2[field]
                        if field_data.ndim > 1:
                            if field_data.shape[1] > 0:
                                field_data = field_data[:, 0]
                            else:
                                field_data = field_data.flatten()
                        solar_components.append(field_data.astype(np.float32))
                    else:
                        print(f"字段 {field} 不存在，用0填充")
                        solar_components.append(np.zeros(len(data), dtype=np.float32))
                solar_data = np.column_stack(solar_components)
                solar_data = solar_data[::12]
                omni_data.append(solar_data)
            except Exception as e:
                print(f"加载失败 {omni_filepath}: {e}")
    
    if all_data:
        all_data = np.concatenate(all_data, axis=0)
        print(f"总数据量: {all_data.shape}")
    else:
        raise ValueError("没有找到数据!")
    
    if omni_data:
        omni_data = np.concatenate(omni_data, axis=0)
        print(f"总数据量: {omni_data.shape}")
    else:
        raise ValueError("没有找到数据!")
    return all_data,omni_data

def get_dataloader(years):
    print("开始加载数据...")
    train_data_all, omni_data = load_all_aurora_data(years)

    # 创建数据集和数据加载器
    print("创建数据集...")
    aurora_dataset = TrainDataset(train_data_all)
    omni_dataset = OmniDataset(omni_data)
    
    aurora_dataloader = DataLoader(
        aurora_dataset, 
        batch_size=4,  # 增加batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # 加速数据传输到GPU
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    omni_dataloader = DataLoader(
        omni_dataset, 
        batch_size=4,  # 增加batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # 加速数据传输到GPU
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    return aurora_dataloader,omni_dataloader