import numpy as np
import pandas as pd
from auroramaps import util as au
from auroramaps import ovation as ao
import math
from datetime import timedelta
import os
import re
import cdflib

core_params = ['Epoch','BX_GSE', 'BY_GSM', 'BZ_GSM','V', 'Pressure']
# 参数合理范围定义
VALID_RANGES = {
    'BX_GSE': (-50, 50),
    'BY_GSM': (-50, 50),      # 典型磁场范围
    'BZ_GSM': (-50, 50),
    'V' : (200,2000), # 合理的太阳风速度范围
    'Pressure': (0.05, 80),  
}

def extract_omni_from_cdf(data_path):
    """从单个cdf文件提取omni数据，返回清理后的DataFrame"""
    # 从文件路径中提取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"处理文件: {file_name}")
    
    try:
        cdf = cdflib.CDF(data_path)
    except Exception as e:
        print(f"加载CDF文件失败 {data_path}: {e}")
        return None
    
    data_dict = {}
    for param in core_params:
        try:
            data_dict[param] = cdf.varget(param)
        except Exception as e:
            print(f"读取参数 {param} 失败: {e}")
            return None
    
    try:
        data_dict['Epoch'] = cdflib.epochs.CDFepoch.to_datetime(data_dict['Epoch'])
    except Exception as e:
        print(f"转换时间戳失败: {e}")
        return None
    
    df = pd.DataFrame(data_dict)
    
    # 记录原始数据行数
    original_rows = len(df)
    print(f"原始数据行数: {original_rows}")
    
    numeric_cols = ['BX_GSE', 'BY_GSM','BZ_GSM','V','Pressure']
    
    # 过滤超出范围的值
    for omni_param in numeric_cols:
        if omni_param in VALID_RANGES:
            min_val, max_val = VALID_RANGES[omni_param]
            original_count = len(df[omni_param])
            
            # 标记超出范围的值
            mask = (df[omni_param] >= min_val) & (df[omni_param] <= max_val)
            invalid_count = np.sum(~mask)
            
            # 将超出范围的值设为NaN
            df.loc[~mask, omni_param] = np.nan
            
            print(f"{omni_param}: 过滤掉 {invalid_count} 个异常点 ({(invalid_count / original_count) * 100:.2f}%)")
    
    # 删除包含NaN值的行
    print(f"删除包含NaN值的行...")
    df_clean = df.dropna(subset=numeric_cols)
    removed_rows = original_rows - len(df_clean)
    print(f"删除了 {removed_rows} 行包含NaN值的数据")
    print(f"清理后剩余行数: {len(df_clean)} ({len(df_clean)/original_rows*100:.2f}%)")
    
    return df_clean

def process_years(start_year=1996, end_year=2016):
    """批量处理指定年份范围的OMNI数据，合并同一年份的上下半年数据"""
    folder_path = "/home/docker/data/ro-share/omni/omni_cdaweb/hourly" 
    output_dir = "/home/docker/data/private/AuroraData/omni_real_data/hourly"
    os.makedirs(output_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"处理 {year} 年数据")
        print('='*60)
        
        year_folder = os.path.join(folder_path, str(year))
        if not os.path.exists(year_folder):
            print(f"警告: {year} 年文件夹不存在: {year_folder}")
            continue
        
        # 查找该年份的所有cdf文件
        cdf_files = []
        for root, dirs, files in os.walk(year_folder):
            for file in files:
                if file.endswith('.cdf'):
                    cdf_files.append(os.path.join(root, file))
        
        if not cdf_files:
            print(f"警告: {year} 年文件夹中没有找到.cdf文件")
            continue
        
        # 按文件名排序，确保顺序正确
        cdf_files.sort()
        print(f"找到 {len(cdf_files)} 个CDF文件:")
        for file in cdf_files:
            print(f"  {os.path.basename(file)}")
        
        # 处理并合并同一年份的所有cdf文件
        all_dfs = []
        
        for cdf_file in cdf_files:
            print(f"\n处理文件: {os.path.basename(cdf_file)}")
            df_clean = extract_omni_from_cdf(cdf_file)
            
            if df_clean is not None and len(df_clean) > 0:
                all_dfs.append(df_clean)
                print(f"成功处理，获得 {len(df_clean)} 行数据")
            else:
                print(f"文件处理失败或没有有效数据")
        
        # 合并同一年份的所有数据
        if all_dfs:
            # 合并所有DataFrame
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # 按时间排序
            combined_df = combined_df.sort_values('Epoch')
            combined_df = combined_df.reset_index(drop=True)
            
            print(f"\n{year} 年数据合并结果:")
            print(f"总数据行数: {len(combined_df)}")
            print(f"时间范围: {combined_df['Epoch'].min()} 到 {combined_df['Epoch'].max()}")
            
            # 检查是否有重复的时间点
            duplicates = combined_df.duplicated(subset=['Epoch'], keep=False)
            if duplicates.any():
                print(f"发现 {duplicates.sum()} 个重复时间点，进行去重...")
                combined_df = combined_df.drop_duplicates(subset=['Epoch'], keep='first')
                print(f"去重后数据行数: {len(combined_df)}")
            
            # 保存合并后的年份数据
            output_file = os.path.join(output_dir, f"omni_{year}_hourly.npy")
            structured_array = combined_df.to_records(index=False)
            
            np.save(output_file, structured_array)
            print(f"已保存: {output_file}")
            print(f"保存数据形状: {structured_array.shape}")


if __name__ == "__main__":
    # 处理1996-2016年的数据
    process_years(start_year=1996, end_year=2016)
    print("\n批量处理完成！")