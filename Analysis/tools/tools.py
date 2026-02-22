

import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import numpy as np
import string
import json
import os
import pandas as pd
import scipy

from scipy.ndimage import gaussian_filter1d

radius = 5.5e-6 #Bead size m
rho = 1.3 * 1000 #Bead density kg/m^3
mu_0 = 4*np.pi*1e-7 #permeability
phase = 5

time_s = 80; time_e = 300
m = 3.5/(10*0.3*3.3)*1e-6

T_0 = 25
r_T_0 = 0.971e3
v = 30000e-6

# our experiments
T = 23
a = 9.2e-4
b = 4.5e-7
r_ = r_T_0/(1+a*(T-T_0)+b*(T-T_0)**2)
nn = v*r_

#Parameters to crop simulation
height = 1544*m #height of FoV in µm
width = 2064*m #width of FoV in µm

start_w =  0#0.0035 - width #flip
end_w = width
start_h = 0.003 - height/2; end_h = 0.003 + height/2


def add_panel_labels(axes, labels=None, xy=(-0.1, 1.1), fontsize=12, weight="bold"):
    ax_list = np.ravel(axes)
    if labels is None:
        labels = list(string.ascii_uppercase[:len(ax_list)])
    for ax, lab in zip(ax_list, labels):
        ax.text(xy[0], xy[1], lab, transform=ax.transAxes,
                fontsize=fontsize, fontweight=weight, va="top", ha="left")

sns.set_theme(style="whitegrid", palette="colorblind")

# Helper: scientific y-axis with math text
def set_sci_y(ax):
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


def func_Fk(x, x_0, H_sat):
    return x/(1/x_0 + np.abs(x)/H_sat)


def procees_track_json(temp_df, times):
    global m, radius, nn, start_h, start_w
    
    temp_df["t"] = times["time"][:-1]

    temp_df["x"] = gaussian_filter1d(temp_df["x"],5)
    temp_df["y"] = gaussian_filter1d(temp_df["y"],5)
    temp_df["x_"] = (temp_df["x"]-temp_df["x"][0])*m
    temp_df["y_"] = (temp_df["y"]-temp_df["y"][0])*m

    temp_df["t"] = (temp_df["t"]-temp_df["t"][0])
    temp_df["y_scaled"] = temp_df["y"]*1e6*m + start_h
    temp_df["x_scaled"] = temp_df["x"]*1e6*m + start_w

    temp_df["r"] = np.sqrt(temp_df["x_"]**2 + temp_df["y_"]**2)
    temp_df["v"] = np.gradient(temp_df["r"], temp_df["t"])
    temp_df["force"] = gaussian_filter1d(6*temp_df["v"].values*radius*np.pi*nn, 10)

    return temp_df

def dw_data(path="./results"):

    def _df_from_columns(obj, cols):
        if isinstance(obj, list):
            return pd.DataFrame(obj)[cols]
        if isinstance(obj, dict):
            arrays = {c: np.asarray(obj.get(c, [])) for c in cols}
            n = min((len(a) for a in arrays.values()), default=0)
            return pd.DataFrame({c: arrays[c][:n] for c in cols})
        return pd.DataFrame()
    
    try:
        arr = np.load(os.path.join(path, "driver.npy")).T
        data_ni = pd.DataFrame(arr, columns=["time", "i_ref", "i_meas", "b_meas", "idx", "feedback"])
    except:
        print("No data_ni data found")
        data_ni = None

    try:
        arr = np.load(os.path.join(path,"timestamps.npy")).T
        data_cam_t =  pd.DataFrame(arr, columns=["time"])
    except:
        print("No data_cam_t data found")
        data_cam_t = None
    try:
        arr = np.load(os.path.join(path,"tracking_data.npy"))
        data_track_t =  pd.DataFrame(arr, columns=["x", "y", "r", "time"])
    except:
        print("No tracking data found")
        data_track_t = None
    
    try:
        with open(os.path.join(path,"track.json"), "rb") as input_file:
            track_json = json.load(input_file)

        df_track = _df_from_columns(track_json["big_0"], ["x", "y"])
        df_track = procees_track_json(df_track, data_cam_t)
    except:
        print("No tracking data found")
        df_track = None

    return data_ni, data_cam_t, data_track_t, df_track




def _compute_derivative(t, y):
    t = t
    y = y

    dy = np.diff(y)
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    d = dy / dt

    if len(d) == 0:
        return np.zeros_like(y)
    
    return np.r_[d[0], d]
    

def estimate_step_delay_xcorr(t, r, m, t_ref):

    window=1.0
    smooth_sigma=3

    max_lag_s=0.6

    mask = (t >= (t_ref - window)) & (t <= (t_ref + window))

    tw = gaussian_filter1d(t[mask], sigma = smooth_sigma)
    rw = gaussian_filter1d(r[mask], sigma = smooth_sigma)
    mw = gaussian_filter1d(m[mask], sigma = smooth_sigma)

    rw = _compute_derivative(tw, rw)
    mw = _compute_derivative(tw, mw)

    # Zero-mean for normalized cross-correlation
    rw_z = rw - np.nanmean(rw)
    mw_z = mw - np.nanmean(mw)

    # Replace NaNs with 0 inside window for correlation stability
    rw_z = np.nan_to_num(rw_z)
    mw_z = np.nan_to_num(mw_z)

    corr = np.correlate(rw_z, mw_z, mode='full')
    lags = np.arange(-len(rw_z) + 1, len(rw_z))

    dt = np.nanmedian(np.diff(tw)) if len(tw) > 1 else 0.0
    if not np.isfinite(dt) or dt <= 0:
        return 0.0

    max_lag_samples = int(max_lag_s / dt)
    center = len(corr) // 2
    lo = max(0, center - max_lag_samples)
    hi = min(len(corr), center + max_lag_samples + 1)

    seg = corr[lo:hi]
    seg_lags = lags[lo:hi]
    best = int(np.nanargmax(seg))
    lag_samples = int(seg_lags[best])

    return float(lag_samples * dt)


def map_times_to_cam_indices(cam_t, t_start, t_end=None):
    idx_start_cam = int(np.searchsorted(cam_t, t_start, side='left'))
    idx_end_cam = None
    if t_end is not None:
        idx_end_cam = int(np.searchsorted(cam_t, t_end, side='left'))
    return idx_start_cam, idx_end_cam