import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
from datetime import datetime, timedelta

error_count = 0

def extract_and_interpolate(file_path):
    global error_count
    try:
        # ----------------------------- 1. 读取数据 -----------------------------
        ds = xr.open_dataset(file_path)
        energy_flux = ds['ENERGY_FLUX_NORTH_MAP'].values  # 形状 (363, 363)
        mlat_orig = ds['LATITUDE_GEOMAGNETIC_GRID_MAP'].values
        mlon_orig = ds['LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP'].values
        avg_time = ds['TIME'].values.item()
        year = int(ds['YEAR'].values.item())
        doy = int(ds['DOY'].values.item())
        avg_time_seconds = float(avg_time)
        base_time = datetime(year-1, 12, 31) + timedelta(days=doy)  # 当年第一天
        data_time = base_time + timedelta(seconds=avg_time_seconds)
        
        # ----------------------------- 2. 展平并筛选有效数据 -----------------------------
        # 将网格展平为一维数组，并过滤掉无效值（NaN）
        valid_mask = ~np.isnan(energy_flux) & ~np.isnan(mlat_orig) & ~np.isnan(mlon_orig)
        mlat_points = mlat_orig[valid_mask]
        mlon_points = mlon_orig[valid_mask]
        flux_points = energy_flux[valid_mask]

        # 将地磁经度 (Mlon) 转换为磁地方时 (MLT)
        mlon_shifted = (mlon_points + 180) % 360 - 180
        mlt_points = (mlon_shifted / 15.0 + 12) % 24
        
        # ----------------------------- 3. 创建规则的80x96目标网格 -----------------------------
        # 磁纬从50到90度，共80个点
        target_mlat = np.linspace(50, 90, 80)
        # 磁地方时从0到24小时，共96个点（不包括24，因为与0等价）
        target_mlt = np.linspace(0.0, 24.0, 96)

        # 生成网格矩阵
        mlat_grid_2d, mlt_grid_2d = np.meshgrid(target_mlat, target_mlt, indexing='ij')
        # 现在 mlat_grid_2d 和 mlt_grid_2d 都是形状为 (80, 96) 的矩阵

        # ----------------------------- 4. 执行插值（非结构点 -> 规则网格）-------------------------
        # 准备源数据点: 每对 (mlat, mlt) 对应一个 flux 值
        points_source = np.column_stack((mlat_points, mlt_points))
        values_source = flux_points

        # 准备目标网格点: 将 (80, 96) 的网格展平，用于插值计算
        points_target = np.column_stack((mlat_grid_2d.ravel(), mlt_grid_2d.ravel()))
        # 使用线性插值。可选项：'linear'， 'nearest'， 'cubic'
        flux_interpolated = griddata(points_source, values_source, points_target,
                                    method='linear', fill_value=np.nan)

        # 将插值结果重塑回 (80, 96) 的二维网格
        flux_grid = flux_interpolated.reshape(mlat_grid_2d.shape)
        flux_grid_filled = np.nan_to_num(flux_grid, nan=0.0)
        
        return flux_grid_filled, data_time
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}, 直接跳过")
        error_count+=1
        return None, None

folder_path = "/home/docker/data/private/AuroraData/real_aurora_data_ssusi/2005"
save_path = "/home/docker/data/private/AuroraData/process_ssusi"

data = {
    'utc':[],
    'aurora_flux':[]
}

os.makedirs(save_path, exist_ok=True)

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.nc'):
            full_path = os.path.join(root, file)
            print(f"找到 .nc 文件: {full_path}")
            flux_grid_filled, data_time = extract_and_interpolate(full_path)
            if flux_grid_filled is None:
                continue
            data['utc'].append(data_time)
            data['aurora_flux'].append(flux_grid_filled)


df = pd.DataFrame(data)
df['utc'] = pd.to_datetime(df['utc'])
df.sort_values(by='utc', inplace=True)
df.reset_index(drop=True, inplace=True)
structured_array = df.to_records(index=False)
np.save(os.path.join(save_path, 'aurora_2005_ssusi.npy'), structured_array)
print("所有文件处理完成, 其中跳过的文件数量:", error_count)
