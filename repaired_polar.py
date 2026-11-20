import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch_npu
from torch.utils.data import DataLoader, Dataset
from models.unet_v1 import UNet
from models.ddpm import DDPM
from data.dataset_diff import OmniDataset
import os
from datetime import datetime
import pandas as pd
from utils.normlize import normalize, denormalize

#---------------------参数设置-----------------------
num_steps = 300
repaint_steps = 10
jump_len = 10
N = 10
n_samples = 1

device = "npu:0"
model_save_path = "/home/docker/code/Aurora_DDPM_final/ckpt/cond/ckptv1_unetv1/aurora_diff_best.pth"
unet = UNet(1, 1)
ddpm = DDPM(unet, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path, map_location=device)
ddpm.load_state_dict(checkpoint['model_state_dict'],strict=False)

#---------------------数据处理-----------------------

data1_path = "/home/docker/data/private/AuroraData/generated_aurora_data/1996_omni_aurora/aurora_img_19960401.npy"
data2_path = "/home/docker/data/private/AuroraData/generated_aurora_data/1996_omni_aurora/aurora_img_19960501.npy"
mn_data1 = np.load(data1_path)
mn_data2 = np.load(data2_path)
data_mn_all = np.concatenate((mn_data1, mn_data2), axis=0)
omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_5min/1996/omni_19960401_5min.npy"
omni_data = np.load(omni_path)
mn_time = omni_data['utc']

polar_data_all = np.load("/home/docker/data/private/AuroraData/real_aurora_data_polar/1996/resampled_5min_1996_0405.npy",allow_pickle=True)
polar_timestamps = polar_data_all['utc']
polar_data = np.stack(polar_data_all['aurora_image'], axis=0).astype(np.float32)

solar_fields = ['Bx', 'By', 'Bz', 'V', 'P']
solar_components = []
for field in solar_fields:
    if field in polar_data_all.dtype.names:
        field_data = polar_data_all[field]
        if field_data.ndim > 1:
            if field_data.shape[1] > 0:
                field_data = field_data[:, 0]
            else:
                field_data = field_data.flatten()
        solar_components.append(field_data.astype(np.float32))
solar_data = np.column_stack(solar_components)
solar_data = OmniDataset(solar_data)

normalizer_real = normalize(polar_data)
denormalizer_real = denormalize(polar_data)


def repair(input_data, mask, solar_point):
    ddpm.eval()
    ddpm.to(device)
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
    repaired_flux = denormalizer_real(repaired_np)
    return repaired_flux



data_dict = {'utc': [], 'image': []}

normalizer_polar = normalize(polar_data)
for i in range(0, len(polar_data)):
    time1 = polar_timestamps[i]
    polar= polar_data[i]
    time1 = polar_timestamps[i].astype('datetime64[s]')
    time1 = pd.Timestamp(time1).to_pydatetime()
    base_time = datetime.fromisoformat("1996-04-01T00:00:00")
    delta = time1 - base_time
    idx_mn = int(delta.total_seconds() // 300) # Assuming 5-minute intervals
    mn_data = data_mn_all[idx_mn]
    
    print("polar time:",time1)
    print("mn_time:", mn_time[idx_mn])
    
    solar_point = solar_data[i]
    mask = np.ones_like(polar)
    real_close_to_zero = polar < 1
    mask[real_close_to_zero] = 0.0
    mask = np.expand_dims(mask, axis=(0, 1))   
     
    polar = normalizer_polar(polar)
    polar = np.expand_dims(polar, axis=(0,1))
    solar_point = solar_point.unsqueeze(0).to(device)
    input_data = polar.copy()
    input_data = torch.tensor(input_data).float().to(device)
    mask = torch.tensor(mask).float().to(device)
    repaired_flux = repair(input_data, mask, solar_point)
    data_dict['utc'].append(time1)
    data_dict['image'].append(repaired_flux)
    
df = pd.DataFrame(
    {
        'utc': data_dict['utc'],
        'image': data_dict['image']
    }
)

save_path ="/home/docker/code/Aurora_DDPM/reasult/polar_res/new_res"
df['utc'] = pd.to_datetime(df['utc'])
df.sort_values(by='utc', inplace=True)
df.reset_index(drop=True, inplace=True)
structured_array = df.to_records(index=False)
np.save(os.path.join(save_path, 'repaired_polar_unetV3_ckptv2.npy'), structured_array)

