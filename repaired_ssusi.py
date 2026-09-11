import torch
import torch.nn as nn
import torch.optim as opt
import torch_npu
from torch.utils.data import DataLoader, Dataset
#from models.simplenet import UNet
from models.unet_v3 import UNet
from models.ddpm import DDPM
import os
from datetime import datetime
import pandas as pd
import numpy as np
from data.dataset_diff import OmniDataset
from utils.normlize import normalize, denormalize
num_steps = 300
repaint_steps = 10
jump_len = 10
N = 10
n_samples = 1

# img_dir = "/home/docker/code/AuroraForecastNet_v1/repaired_aurora_imgs_ssusi_11/"
# os.makedirs(img_dir, exist_ok=True)

device = "npu:0"
model_save_path = "/home/docker/code/Aurora_DDPM/ckpt/diffusion_ckpt_unet/ckpt_v2_unetv3/aurora_diff_best.pth"
unet = UNet(1, 1)
ddpm = DDPM(unet, num_train_steps=1000, schedule='cosine')
checkpoint = torch.load(model_save_path, map_location=device)
ddpm.load_state_dict(checkpoint['model_state_dict'],strict=False)
#ddpm.load_state_dict(checkpoint)


data_path = "/home/docker/data/private/AuroraData/generated_aurora_data/2005_omni_aurora/aurora_img_20050101.npy"
data_mn_all = np.load(data_path)
omni_path = "/home/docker/data/private/AuroraData/omni_real_data/omni_1min_pro/2005/omni_20050101_1min.npy"
omni_data = np.load(omni_path)
mn_time = omni_data['utc']
ssusi_data = np.load('/home/docker/data/private/AuroraData/process_ssusi/aurora_2005_ssusi.npy', allow_pickle=True)
aurora_data_ssusi = np.stack(ssusi_data['aurora_flux'], axis=0).astype(np.float32)
ssusi_timestamps = ssusi_data['utc']


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
solar_data = np.column_stack(solar_components)
solar_data = OmniDataset(solar_data)

normalizer_real = normalize(aurora_data_ssusi)
denormalizer_real = denormalize(aurora_data_ssusi)



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


polar_dir = "/home/docker/code/Aurora_DDPM/reasult/polar_res/new_res/repaired_aurora_imgs_polar_unetv3_ckptv0/"
os.makedirs(polar_dir, exist_ok=True)

data_dict = {'utc': [], 'image': []}
for i in range(0, 400):
    ssusi1= aurora_data_ssusi[i]
    time1 = ssusi_timestamps[i].astype('datetime64[s]')
    time1 = pd.Timestamp(time1).to_pydatetime()
    base_time = datetime.fromisoformat("2005-01-01T00:00:00")
    delta = time1 - base_time
    idx_mn = int(delta.total_seconds() // 60 ) # Assuming 5-minute intervals
    mn_data = data_mn_all[idx_mn]
    solar_point = solar_data[idx_mn]
    
    print("ssusi:",time1)
    print("omni_time:",mn_time[idx_mn])
    
    mask = np.ones_like(ssusi1)
    real_close_to_zero = ssusi1 < 1
    sim_has_value = mn_data > 0
    need_repair = real_close_to_zero & sim_has_value
    mask[need_repair] = 0.0
    mask = np.expand_dims(mask, axis=(0, 1))   
    
    ssusi1 = normalizer_real(ssusi1)
    ssusi1 = np.expand_dims(ssusi1, axis=(0,1))
    input_data = ssusi1.copy()
    input_data = torch.tensor(input_data).float().to(device)
    mask = torch.tensor(mask).float().to(device)
    solar_point = solar_point.unsqueeze(0).to(device)
    
    repaired_flux = repair(input_data, mask, solar_point)
    data_dict['utc'].append(time1)
    data_dict['image'].append(repaired_flux)
    
df = pd.DataFrame(
    {
        'utc': data_dict['utc'],
        'image': data_dict['image']
    }
)

save_path ="/home/docker/code/Aurora_DDPM/reasult/dmsp_res/new_res"
df['utc'] = pd.to_datetime(df['utc'])
df.sort_values(by='utc', inplace=True)
df.reset_index(drop=True, inplace=True)
structured_array = df.to_records(index=False)
np.save(os.path.join(save_path, 'repaired_ssusi_unetV3_ckptv2.npy'), structured_array) 