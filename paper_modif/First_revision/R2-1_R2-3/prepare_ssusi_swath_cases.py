import os
import glob
import json
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imread

import scipy.interpolate
import scipy.ndimage

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade

import aacgmv2


# =====================================================================
# 路径与全局配置
# =====================================================================
RAW_EDR_ROOT = "/home/docker/data/private/AuroraData/real_aurora_data_ssusi/f16/2005"
OUTPUT_DIR = "/home/docker/code/Aurora_DDPM_total/paper_modif/First_revision/R2-1_R2-3/swath_prepared_cases"

BACKGROUND_IMG = "/home/docker/data/private/AuroraData/background_img/natural-earth-1_large2048px.png"

# Figure 6 / Figure 7 的论文目标时间
TARGET_CASES = {
    "Figure6": datetime(2005, 1, 1, 15, 21, 0),
    "Figure7": datetime(2005, 1, 4, 3, 3, 0),
}

# 目标网格（与你论文中 NH grid 保持一致）
MLAT_1D_POS = np.linspace(50.0, 90.0, 80)
MLT_1D = np.linspace(0.0, 24.0, 96, endpoint=False)   # 注意 endpoint=False，避免 0/24 重复

# 视觉参数
ORTHO_LON = 110.0
VMAX = 5.0


# =====================================================================
# NOAA 风格 colormap
# =====================================================================
def make_aurora_cmap():
    colors = [
        (0.0, 0.2, 0.0), (0.0, 0.5, 0.0), (0.0, 0.8, 0.0),
        (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.6, 0.0),
        (1.0, 0.3, 0.0), (1.0, 0.0, 0.0),
    ]
    cm = LinearSegmentedColormap.from_list('aurora', colors, N=256)
    cm.set_bad(color='white', alpha=0.0)
    return cm


AURORA_CMAP = make_aurora_cmap()


# =====================================================================
# 时间解析
# =====================================================================
def parse_time_basic(s):
    """
    支持:
      2005-01-01T14:44:34
      2005001T144434
    """
    s = s.strip()
    if "-" in s and ":" in s:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass

    # 例如 2005001T144434
    try:
        year = int(s[:4])
        doy = int(s[4:7])
        hour = int(s[8:10])
        minute = int(s[10:12])
        second = int(s[12:14])
        return datetime(year, 1, 1) + timedelta(days=doy - 1,
                                                hours=hour,
                                                minutes=minute,
                                                seconds=second)
    except Exception:
        raise ValueError(f"Unsupported time string: {s}")


def find_all_edr_files(root):
    return sorted(glob.glob(os.path.join(root, "*", "*.nc")))


def infer_interval_from_filename(path):
    """
    文件名示例:
    dmspf16_ssusi_edr-aurora_2005001T144434-2005001T162625-REV06228_vA8.2.0r000.nc
    """
    base = os.path.basename(path)
    parts = base.split("_")
    last = parts[-2] if len(parts) >= 2 else base
    # 实际 interval 在 aurora_ 后面
    try:
        after = base.split("aurora_")[1]
        interval_part = after.split("-REV")[0]
        t0_str, t1_str = interval_part.split("-")
        t0 = parse_time_basic(t0_str)
        t1 = parse_time_basic(t1_str)
        return t0, t1
    except Exception:
        return None, None


def find_nearest_edr(target_time, root):
    files = find_all_edr_files(root)
    best = None
    best_dt = None
    best_mid = None

    for fp in files:
        t0, t1 = infer_interval_from_filename(fp)
        if t0 is None or t1 is None:
            continue
        mid = t0 + (t1 - t0) / 2
        dt = abs((mid - target_time).total_seconds())
        if (best is None) or (dt < best_dt):
            best = fp
            best_dt = dt
            best_mid = mid

    if best is None:
        raise RuntimeError(f"No EDR files found under {root}")
    return best, best_mid


# =====================================================================
# NetCDF 变量读取
# =====================================================================
def safe_read_var(ds, name):
    if name not in ds.variables:
        raise KeyError(f"Variable {name} not found.")
    arr = ds.variables[name][:]
    try:
        arr = np.array(arr)
    except Exception:
        arr = np.asarray(arr)
    return arr


def decode_scalar_attr(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.size == 1:
        return decode_scalar_attr(x.item())
    return x


def load_ssusi_edr_case(nc_path):
    ds = Dataset(nc_path, "r")

    flux = safe_read_var(ds, "ENERGY_FLUX_NORTH_MAP")
    mlat = safe_read_var(ds, "LATITUDE_GEOMAGNETIC_GRID_MAP")
    mlt = safe_read_var(ds, "MLT_GRID_MAP")
    ut_n = safe_read_var(ds, "UT_N")

    flux = np.array(flux, dtype=np.float64)
    mlat = np.array(mlat, dtype=np.float64)
    mlt = np.array(mlt, dtype=np.float64)
    ut_n = np.array(ut_n, dtype=np.float64)

    # 真正的观测支持：UT_N > 0
    obs_native = np.isfinite(ut_n) & (ut_n > 0)

    # flux 过滤
    valid_native = obs_native & np.isfinite(flux) & np.isfinite(mlat) & np.isfinite(mlt)

    # metadata
    meta = {}
    for key in [
        "STARTING_TIME", "STOPPING_TIME",
        "STARTING_ORBIT_NUMBER", "STOPPING_ORBIT_NUMBER",
        "DATA_DESCRIPTION", "DATA_VERSION"
    ]:
        if hasattr(ds, key):
            meta[key] = decode_scalar_attr(getattr(ds, key))

    ds.close()
    return flux, mlat, mlt, ut_n, valid_native, meta


# =====================================================================
# 目标网格边界
# =====================================================================
def centers_to_edges(x):
    x = np.asarray(x)
    dx = np.diff(x)
    edges = np.zeros(len(x) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


MLAT_EDGES = centers_to_edges(MLAT_1D_POS)
MLT_STEP = 24.0 / len(MLT_1D)
MLT_EDGES = np.arange(0.0, 24.0 + MLT_STEP, MLT_STEP)


# =====================================================================
# direct-support 分箱
# =====================================================================
def bin_flux_direct(flux, mlat, mlt, valid_native):
    """
    直接把 native swath-supported pixels 分箱到 80x96 目标网格。
    不做“把空区插值成观测”的事情。
    """
    flux_valid = flux[valid_native]
    mlat_valid = mlat[valid_native]
    mlt_valid = np.mod(mlt[valid_native], 24.0)

    # 过滤到 NH grid 范围
    keep = (
        np.isfinite(flux_valid) &
        np.isfinite(mlat_valid) &
        np.isfinite(mlt_valid) &
        (mlat_valid >= 50.0) & (mlat_valid <= 90.0)
    )
    flux_valid = flux_valid[keep]
    mlat_valid = mlat_valid[keep]
    mlt_valid = mlt_valid[keep]

    i_lat = np.digitize(mlat_valid, MLAT_EDGES) - 1
    i_mlt = np.digitize(mlt_valid, MLT_EDGES) - 1

    # 修正 mlt=24 的边界（虽然 endpoint=False 基本不会遇到）
    i_mlt[i_mlt == len(MLT_1D)] = 0

    inside = (
        (i_lat >= 0) & (i_lat < len(MLAT_1D_POS)) &
        (i_mlt >= 0) & (i_mlt < len(MLT_1D))
    )
    flux_valid = flux_valid[inside]
    i_lat = i_lat[inside]
    i_mlt = i_mlt[inside]

    sum_grid = np.zeros((len(MLAT_1D_POS), len(MLT_1D)), dtype=np.float64)
    cnt_grid = np.zeros((len(MLAT_1D_POS), len(MLT_1D)), dtype=np.int32)

    for f, ilat, imlt in zip(flux_valid, i_lat, i_mlt):
        sum_grid[ilat, imlt] += f
        cnt_grid[ilat, imlt] += 1

    flux_grid = np.full_like(sum_grid, np.nan, dtype=np.float64)
    obs_mask = cnt_grid > 0
    flux_grid[obs_mask] = sum_grid[obs_mask] / cnt_grid[obs_mask]

    return flux_grid.astype(np.float32), obs_mask.astype(np.uint8), cnt_grid


# =====================================================================
# MLAT-MLT 诊断图
# =====================================================================
def plot_mlat_mlt_case(case_name, flux_grid, obs_mask, count_grid, save_path, title_extra=""):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)

    extent = [0.0, 24.0, 50.0, 90.0]

    im0 = axes[0].imshow(
        np.where(np.isfinite(flux_grid), flux_grid, np.nan),
        origin="lower", aspect="auto", extent=extent,
        cmap=AURORA_CMAP, vmin=0, vmax=VMAX
    )
    axes[0].set_title("Swath-aware direct-binned flux")
    axes[0].set_xlabel("MLT [h]")
    axes[0].set_ylabel("MLAT [deg]")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(r"Energy Flux (ergs cm$^{-2}$ s$^{-1}$)")

    im1 = axes[1].imshow(
        obs_mask.astype(float), origin="lower", aspect="auto", extent=extent,
        cmap="viridis", vmin=0, vmax=1
    )
    axes[1].set_title("Observation mask (UT_N > 0 propagated)")
    axes[1].set_xlabel("MLT [h]")
    axes[1].set_ylabel("MLAT [deg]")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Observed / Missing")

    im2 = axes[2].imshow(
        count_grid.astype(float), origin="lower", aspect="auto", extent=extent,
        cmap="magma"
    )
    axes[2].set_title("Native sample count per target cell")
    axes[2].set_xlabel("MLT [h]")
    axes[2].set_ylabel("MLAT [deg]")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Count")

    fig.suptitle(f"{case_name} — {title_extra}", fontsize=14)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =====================================================================
# 正射投影绘图（参考你给的代码）
# =====================================================================
def plot_ovation_style(timestamp, flux_2d, mlat_1d_signed, mlt_1d,
                       hemisphere, save_path, title_prefix):
    """
    AACGM → geographic → orthographic projection
    """
    mltN, mlatN = np.meshgrid(mlt_1d, mlat_1d_signed)

    # MLT -> AACGM longitude
    mlon_1d_small = aacgmv2.convert_mlt(mltN[0], timestamp, m2a=True)
    mlon_1d = np.tile(mlon_1d_small, mlatN.shape[0])
    mlat_1d_flat = np.squeeze(mlatN.reshape(np.size(mltN), 1))

    glat_1d, glon_1d, _ = aacgmv2.convert_latlon_arr(
        mlat_1d_flat, mlon_1d, 100, timestamp, method_code="A2G"
    )

    geo_2d = np.vstack((glat_1d, glon_1d)).T
    flux_1d = flux_2d.reshape(np.size(mltN), 1)

    h, w = 512, 1024
    wx, wy = np.mgrid[-90:90:180/h, -180:180:360/w]

    aimg = np.squeeze(scipy.interpolate.griddata(
        geo_2d, flux_1d, (wx, wy), method='linear', fill_value=0
    ))
    aimg = scipy.ndimage.gaussian_filter(aimg, sigma=(2, 3), mode='wrap')

    aimg[aimg < 0.1] = np.nan
    aimg = aimg.astype(np.float32)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    fig.set_facecolor('black')

    center_lat = 90.0 if hemisphere == 'N' else -90.0
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.Orthographic(ORTHO_LON, center_lat))

    if os.path.exists(BACKGROUND_IMG):
        ax.imshow(imread(BACKGROUND_IMG), origin='upper',
                  transform=ccrs.PlateCarree(),
                  extent=[-180, 180, -90, 90], zorder=0)
    else:
        ax.stock_img()

    ax.gridlines(linestyle='dashed', alpha=0.3, color='white')
    ax.coastlines('50m', color='white', alpha=0.5, linewidth=0.5)
    ax.add_feature(Nightshade(timestamp, alpha=0.4))

    img = ax.imshow(aimg, vmin=0, vmax=VMAX,
                    transform=ccrs.PlateCarree(),
                    extent=[-180, 180, -90, 90], origin='lower',
                    zorder=3, alpha=0.85, cmap=AURORA_CMAP)

    time_str = timestamp.strftime("%Y-%m-%d %H:%M UT")
    fig.text(0.5, 0.92, f'{title_prefix} ({hemisphere}H) - {time_str}',
             color='white', fontsize=18, ha='center', weight='bold')

    cbax = fig.add_axes([0.3, 0.08, 0.4, 0.02])
    cbar = plt.colorbar(img, cax=cbax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12, colors='white')
    cbar.set_label(r'Energy Flux (ergs cm$^{-2}$ s$^{-1}$)',
                   color='white', fontsize=14)

    plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
    plt.close()


# =====================================================================
# 保存 metadata
# =====================================================================
def save_metadata_json(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


# =====================================================================
# 主流程
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}

    for case_name, target_time in TARGET_CASES.items():
        print("=" * 80)
        print(case_name, target_time)

        nc_path, midpoint = find_nearest_edr(target_time, RAW_EDR_ROOT)
        print("Nearest file:", nc_path)
        print("Midpoint:", midpoint)

        t0, t1 = infer_interval_from_filename(nc_path)

        flux, mlat, mlt, ut_n, valid_native, meta_nc = load_ssusi_edr_case(nc_path)

        flux_grid, obs_mask, count_grid = bin_flux_direct(flux, mlat, mlt, valid_native)

        # 简单统计
        obs_frac = float(obs_mask.mean())
        mean_flux = float(np.nanmean(flux_grid)) if np.any(np.isfinite(flux_grid)) else np.nan
        max_flux = float(np.nanmax(flux_grid)) if np.any(np.isfinite(flux_grid)) else np.nan

        case_dir = os.path.join(OUTPUT_DIR, case_name)
        os.makedirs(case_dir, exist_ok=True)

        npz_path = os.path.join(case_dir, f"{case_name}_swath_case.npz")
        np.savez_compressed(
            npz_path,
            flux_grid=flux_grid,
            obs_mask=obs_mask,
            count_grid=count_grid,
            mlat_1d=MLAT_1D_POS,
            mlt_1d=MLT_1D
        )

        meta = {
            "case_name": case_name,
            "nominal_manuscript_time": target_time.isoformat(),
            "matched_edr_file": nc_path,
            "edr_midpoint": midpoint.isoformat(),
            "acquisition_start": t0.isoformat() if t0 else None,
            "acquisition_end": t1.isoformat() if t1 else None,
            "obs_fraction": obs_frac,
            "mean_flux": mean_flux,
            "max_flux": max_flux,
            "netcdf_metadata": meta_nc,
        }
        save_metadata_json(os.path.join(case_dir, f"{case_name}_metadata.json"), meta)

        # 诊断图
        diag_path = os.path.join(case_dir, f"{case_name}_mlat_mlt_diagnostic.png")
        plot_mlat_mlt_case(
            case_name=case_name,
            flux_grid=flux_grid,
            obs_mask=obs_mask,
            count_grid=count_grid,
            save_path=diag_path,
            title_extra=f"{midpoint.strftime('%Y-%m-%d %H:%M:%S')} UT"
        )

        # orthographic 图：只在 observed 区域绘制
        flux_for_plot = flux_grid.copy()
        flux_for_plot[obs_mask == 0] = np.nan

        ortho_path = os.path.join(case_dir, f"{case_name}_orthographic.png")
        plot_ovation_style(
            timestamp=midpoint,
            flux_2d=flux_for_plot,
            mlat_1d_signed=MLAT_1D_POS,
            mlt_1d=MLT_1D,
            hemisphere='N',
            save_path=ortho_path,
            title_prefix=f"{case_name} SSUSI Auroral EDR"
        )

        summary[case_name] = meta

        print(f"Saved case npz: {npz_path}")
        print(f"Saved metadata: {os.path.join(case_dir, f'{case_name}_metadata.json')}")
        print(f"Saved diagnostic: {diag_path}")
        print(f"Saved orthographic: {ortho_path}")
        print(f"Observed fraction: {obs_frac:.4f}")
        print(f"Mean flux: {mean_flux:.4f}")
        print(f"Max flux: {max_flux:.4f}")

    save_metadata_json(os.path.join(OUTPUT_DIR, "summary.json"), summary)

    print("=" * 80)
    print("DONE")
    print("Output dir:", OUTPUT_DIR)


if __name__ == "__main__":
    main()