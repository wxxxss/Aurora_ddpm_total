import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class AuroraDataset(Dataset):
    def __init__(self, aurora_data, omni_data, split='train', train_ratio=0.8, 
                 norm_params=None, is_train=True):
        """
        整合的极光和OMNI数据集
        
        参数:
            aurora_data: 极光数据
            omni_data: OMNI数据
            split: 'train' 或 'val'
            train_ratio: 训练集比例
            norm_params: 归一化参数字典，如果为None则从训练数据计算
            is_train: 是否为训练模式（影响mask生成策略）
        """
        assert len(aurora_data) == len(omni_data), "极光和OMNI数据长度不一致"
        
        self.aurora_data = aurora_data
        self.omni_data = omni_data
        self.split = split
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.num_masks_per_sample = 20
        
        # 划分数据集
        n_total = len(aurora_data)
        n_train = int(n_total * train_ratio)
        
        if split == 'train':
            self.aurora_split = aurora_data[:n_train]
            self.omni_split = omni_data[:n_train]
            self.indices = np.arange(n_train)
        elif split == 'val':
            self.aurora_split = aurora_data[n_train:]
            self.omni_split = omni_data[n_train:]
            self.indices = np.arange(n_total - n_train)
        
        # 归一化处理
        if norm_params is None:
            # 训练集：计算归一化参数
            self.norm_params = self._calculate_norm_params()
        else:
            # 验证集：使用传入的归一化参数
            self.norm_params = norm_params
        
        # 对极光数据应用log变换并归一化
        self.aurora_split_normalized = self._normalize_aurora(self.aurora_split)
        
        # 对OMNI数据归一化
        self.omni_split_normalized = self._normalize_omni(self.omni_split)
        
    
    def _calculate_norm_params(self):
        """计算归一化参数"""
        # 极光数据的log变换参数
        log_aurora = np.log1p(self.aurora_split)
        aurora_min = float(log_aurora.min())
        aurora_max = float(log_aurora.max())
        aurora_range = max(aurora_max - aurora_min, 1e-6)
        
        # OMNI数据的归一化参数
        omni_min = self.omni_split.min(axis=0)
        omni_max = self.omni_split.max(axis=0)
        omni_range = omni_max - omni_min
        omni_range[omni_range == 0] = 1.0  # 避免除零
        
        norm_params = {
            'aurora': {
                'min': aurora_min,
                'max': aurora_max,
                'range': aurora_range
            },
            'omni': {
                'min': omni_min,
                'max': omni_max,
                'range': omni_range
            }
        }
        
        return norm_params
    
    def _normalize_aurora(self, aurora_data):
        """归一化极光数据"""
        normalized = []
        for data in aurora_data:
            # 应用log变换
            log_data = np.log1p(data)
            # 归一化到[0, 1]
            norm_data = (log_data - self.norm_params['aurora']['min']) / self.norm_params['aurora']['range']
            norm_data = np.clip(norm_data, 0.0, 1.0).astype(np.float32)
            normalized.append(norm_data)
        
        return np.array(normalized)
    
    def _normalize_omni(self, omni_data):
        """归一化OMNI数据"""
        # 归一化到[0, 1]
        norm_data = (omni_data - self.norm_params['omni']['min']) / self.norm_params['omni']['range']
        norm_data = np.clip(norm_data, 0.0, 1.0).astype(np.float32)
        return norm_data
    
    def __len__(self):
        return len(self.aurora_split_normalized)
    
    def __getitem__(self, idx):
        # 获取极光数据
        aurora_data = torch.from_numpy(self.aurora_split_normalized[idx]).float()
        aurora_data = aurora_data.unsqueeze(0)  # 形状: (1, H, W)
        
        # 获取OMNI数据
        omni_data = torch.from_numpy(self.omni_split_normalized[idx]).float()
        
        return aurora_data, omni_data
    
    def save_norm_params(self, filepath):
        """保存归一化参数到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.norm_params, f)
        print(f"归一化参数已保存到: {filepath}")
    
    @classmethod
    def load_norm_params(cls, filepath):
        """从文件加载归一化参数"""
        with open(filepath, 'rb') as f:
            norm_params = pickle.load(f)
        print(f"归一化参数已从 {filepath} 加载")
        return norm_params

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


def load_all_aurora_data(years=[1996, 1997, 1998], months= range(1, 13)):
    """加载多个月份的数据，并间隔采样（每小时一个点）"""
    #solar_fields = ['Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'P']
    solar_fields = ['Bx', 'By', 'Bz', 'V', 'P']
    all_aurora_data = []
    all_omni_data = []
    
    base_path = "/home/docker/data/private/AuroraData/generated_aurora_data"
    omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_5min"
    
    for year in years:
        year_path = os.path.join(base_path, f"{year}_omni_aurora")
        omni_year_path = os.path.join(omni_path, f"{year}")
        
        if not os.path.exists(year_path):
            continue
            
        for month in months:
            # 加载极光数据
            filename = f"aurora_img_{year}{month:02d}01.npy"
            filepath = os.path.join(year_path, filename)
            
            # 加载OMNI数据
            omni_filename = f"omni_{year}{month:02d}01_5min.npy"
            omni_filepath = os.path.join(omni_year_path, omni_filename)
            
            try:
                # 处理极光数据
                aurora_data = np.load(filepath)
                #aurora_data_sampled = aurora_data[::12]  # 每小时一个点
                aurora_data_sampled = aurora_data
                all_aurora_data.append(aurora_data_sampled)
                print(f"加载极光数据: {filename}, 形状: {aurora_data_sampled.shape}")
            except Exception as e:
                print(f"加载极光数据失败 {filename}: {e}")
            
            try:
                # 处理OMNI数据
                omni_raw_data = np.load(omni_filepath, allow_pickle=True)
                solar_components = []
                
                for field in solar_fields:
                    if field in omni_raw_data.dtype.names:
                        field_data = omni_raw_data[field]
                        if field_data.ndim > 1:
                            if field_data.shape[1] > 0:
                                field_data = field_data[:, 0]
                            else:
                                field_data = field_data.flatten()
                        solar_components.append(field_data.astype(np.float32))
                    else:
                        print(f"字段 {field} 不存在，用0填充")
                        solar_components.append(np.zeros(len(omni_raw_data), dtype=np.float32))
                
                omni_data = np.column_stack(solar_components)
                #omni_data_sampled = omni_data[::12]  # 与极光数据同步采样
                omni_data_sampled = omni_data
                all_omni_data.append(omni_data_sampled)
                print(f"加载OMNI数据: {omni_filename}, 形状: {omni_data_sampled.shape}")
            except Exception as e:
                print(f"加载OMNI数据失败 {omni_filename}: {e}")
    
    # 合并所有数据
    if all_aurora_data and all_omni_data:
        aurora_data_concat = np.concatenate(all_aurora_data, axis=0)
        omni_data_concat = np.concatenate(all_omni_data, axis=0)
        
        print(f"极光数据总形状: {aurora_data_concat.shape}")
        print(f"OMNI数据总形状: {omni_data_concat.shape}")
        
        # 确保数据长度一致
        min_len = min(len(aurora_data_concat), len(omni_data_concat))
        aurora_data_concat = aurora_data_concat[:min_len]
        omni_data_concat = omni_data_concat[:min_len]
        
        print(f"对齐后极光数据形状: {aurora_data_concat.shape}")
        print(f"对齐后OMNI数据形状: {omni_data_concat.shape}")
        
        return aurora_data_concat, omni_data_concat
    else:
        raise ValueError("没有找到有效的数据!")


def get_dataloaders(years=[1996, 1997, 1998], months= range(1, 13),batch_size=4, train_ratio=0.8, 
                    save_norm_params=False, norm_params_path='norm_params.pkl'):
    """
    获取训练集和验证集的数据加载器
    
    参数:
        years: 年份列表
        months: 月份列表
        batch_size: 批次大小
        train_ratio: 训练集比例
        save_norm_params: 是否保存归一化参数
        norm_params_path: 归一化参数保存路径
    
    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        train_dataset.norm_params: 归一化参数
    """
    print("开始加载数据...")
    aurora_data, omni_data = load_all_aurora_data(years, months)
    # mn_max = aurora_data.max()
    # mn_min = aurora_data.min()
    # polar_max=39.999454
    # polar_min=0.0
    # aurora_data = (aurora_data - mn_min) / (mn_max - mn_min) * (polar_max - polar_min) + polar_min
    
    print(f"\n数据统计:")
    print(f"极光数据: {aurora_data.shape}, 范围: [{aurora_data.min():.4f}, {aurora_data.max():.4f}]")
    print(f"OMNI数据: {omni_data.shape}, 范围: [{omni_data.min(axis=0)}, {omni_data.max(axis=0)}]")
    
    # 创建训练集
    print("\n创建训练集...")
    train_dataset = AuroraDataset(
        aurora_data, omni_data, 
        split='train', 
        train_ratio=train_ratio,
        norm_params=None,  # 训练集计算归一化参数
        is_train=True
    )
    
    # 保存归一化参数
    if save_norm_params:
        train_dataset.save_norm_params(norm_params_path)
    
    # 创建验证集（使用训练集的归一化参数）
    print("\n创建验证集...")
    val_dataset = AuroraDataset(
        aurora_data, omni_data,
        split='val',
        train_ratio=train_ratio,
        norm_params=train_dataset.norm_params,  # 使用训练集的归一化参数
        is_train=False  # 验证模式
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        num_workers=4,
        pin_memory=True,
        drop_last=False  # 验证集不需要drop_last
    )
    
    print(f"\n数据集统计:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"总数据集大小: {len(train_dataset) + len(val_dataset)}")
    print(f"训练集比例: {len(train_dataset) / (len(train_dataset) + len(val_dataset)):.2%}")
    
    return train_loader, val_loader

def get_dataloaders_combine(data_path, batch_size=4, train_ratio=0.8, 
                    save_norm_params=False, norm_params_path='norm_params.pkl'):
    """
    获取训练集和验证集的数据加载器
    
    参数:
        years: 年份列表
        batch_size: 批次大小
        train_ratio: 训练集比例
        save_norm_params: 是否保存归一化参数
        norm_params_path: 归一化参数保存路径
    
    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        train_dataset.norm_params: 归一化参数
    """
    print("开始加载数据...")
    data_all = np.load(data_path, allow_pickle=True)
    aurora_data = np.stack(data_all['aurora_image'], axis=0).astype(np.float32)
    omni_fields = ['BX_GSE', 'BY_GSM','BZ_GSM','V','Pressure']
    omni_data = np.stack([data_all[field] for field in omni_fields], axis=1).astype(np.float32)
    
    
    print(f"\n数据统计:")
    print(f"极光数据: {aurora_data.shape}, 范围: [{aurora_data.min():.4f}, {aurora_data.max():.4f}]")
    print(f"OMNI数据: {omni_data.shape}, 范围: [{omni_data.min(axis=0)}, {omni_data.max(axis=0)}]")
    
    # 创建训练集
    print("\n创建训练集...")
    train_dataset = AuroraDataset(
        aurora_data, omni_data, 
        split='train', 
        train_ratio=train_ratio,
        norm_params=None,  # 训练集计算归一化参数
        is_train=True
    )
    
    # 保存归一化参数
    if save_norm_params:
        train_dataset.save_norm_params(norm_params_path)
    
    # 创建验证集（使用训练集的归一化参数）
    print("\n创建验证集...")
    val_dataset = AuroraDataset(
        aurora_data, omni_data,
        split='val',
        train_ratio=train_ratio,
        norm_params=train_dataset.norm_params,  # 使用训练集的归一化参数
        is_train=False  # 验证模式
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        num_workers=4,
        pin_memory=True,
        drop_last=False  # 验证集不需要drop_last
    )
    
    print(f"\n数据集统计:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"总数据集大小: {len(train_dataset) + len(val_dataset)}")
    print(f"训练集比例: {len(train_dataset) / (len(train_dataset) + len(val_dataset)):.2%}")
    
    return train_loader, val_loader

