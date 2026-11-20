import torch
import torch.nn as nn
import numpy as np
class normalize(nn.Module):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas
        log_data = np.log1p(self.datas)
        self.log_min = float(log_data.min())
        self.log_max = float(log_data.max())
        self.log_range = max(self.log_max - self.log_min, 1e-6)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.log1p(x)
        x = (x - self.log_min) / self.log_range
        x = np.clip(x, 0.0, 1.0)
        # return (x * 2.0 - 1.0).astype(np.float32)
        return x.astype(np.float32)
    
class denormalize(nn.Module):
    def __init__(self, datas):
        super().__init__()
        self.datas = np.array(datas, dtype=np.float32) 
        log_data = np.log1p(self.datas)
        self.log_min = float(log_data.min())
        self.log_max = float(log_data.max())
        self.log_range = max(self.log_max - self.log_min, 1e-6)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # aurora_01 = (x + 1.0) / 2.0
        aurora_01 = x
        aurora_log = aurora_01 * self.log_range + self.log_min
        aurora_flux = np.expm1(aurora_log)
        return aurora_flux