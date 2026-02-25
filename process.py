#!/usr/bin/env python

import os
import glob
import matplotlib as m
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from pprint import pprint
import pathlib
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import cmcrameri.cm as cmc

__author__ = "Thierry Slot"

logging.getLogger().setLevel(logging.CRITICAL)

force_reprocessing = False


# =====================================================================================
# Helpers
# =====================================================================================

def ensure_dir(file_path):
    """
    Ensure that the directory for file_path exists.
    Returns file_path unchanged.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")


def generate_ECSA_figure(data, title="CV", axisX=[0, 0], axisY=[0, 0], box_text=""):

    w, h = 15, 10
    fig = plt.figure(figsize=(w, h))
    axes = plt.gca()
    axes.xaxis.grid(color='lightgrey', linestyle='dotted',
                    linewidth=2)  # vertical lines
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)

    if (axisX[0] != axisX[1]):
        axes.set_xlim(axisX)
    if (axisY[0] != axisY[1]):
        axes.set_ylim(axisY)

    plt.plot(data[0][1], data[0][2], label=data[0][0],
             linewidth=0, marker="o", color="black")
    plt.plot(data[1][1], data[1][2], label=data[1]
             [0], linewidth=2, marker="", color="red")

    plt.title(title)
    plt.legend(loc='lower right')
    if box_text != "":
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
        axes.text(0.98, 1 - (0.02*w/h), box_text, transform=axes.transAxes, fontsize=22,
                  verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.rcParams.update({'font.size': 22})
    return fig


def generate_cv_figure(data, title="CV", axisX=[0, 0], axisY=[0, 0], box_text=""):

    w, h = 15, 10
    fig = plt.figure(figsize=(w, h))

    axes = plt.gca()
    axes.xaxis.grid(color='lightgrey', linestyle='dotted',
                    linewidth=2)  # vertical lines
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)

    if (axisX[0] != axisX[1]):
        axes.set_xlim(axisX)
    if (axisY[0] != axisY[1]):
        axes.set_ylim(axisY)

    d_colors = ['blue', 'red', 'green', 'black',
                'purple', 'organge', 'brown', 'magenta']
    data_cnt = 0
    color_cnt = 0
    for subdata in data:
        color_cnt = data_cnt % len(d_colors) + 1
        data_cnt += 1
        label = subdata[0]
        sel_x_red = subdata[1]
        sel_y_red = subdata[2]
        plt.plot(sel_x_red, sel_y_red, label=label,
                 linewidth=2, color=d_colors[color_cnt-1])

    plt.title(title)
    plt.legend(loc='lower right')
    if box_text != "":
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
        axes.text(0.98, 1 - (0.02*w/h), box_text, transform=axes.transAxes, fontsize=22,
                  verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.rcParams.update({'font.size': 22})
    return fig


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def round_nearest(x, a):
    return round(x / a) * a


def plotContour(df0, x_multiplier=1, section_x=[-1.6, -1.09], section_y=[0, 150],
                section_z=None, z_sublevels=8, z_colorlevels=256, z_alpha=1,
                w=15, h=10, scaling_factor=0.7, AxesOff=False):
    # SECTIONING

    # Convert the read JSON data from STR into FLOAT values
    df0_x = [float(x) for x in df0.columns.values.tolist()]

    x_ix = [find_nearest(df0_x, section_x[0]), find_nearest(df0_x, section_x[1])]
    y_ix = section_y

    # subselection of the data
    df0 = df0.iloc[y_ix[0]: y_ix[1], x_ix[0]: x_ix[1]].copy()

    # PLOTTING
    scaling_factor = 0.7
    font_scaling = scaling_factor / 0.7
    w, h = scaling_factor*w, scaling_factor*h

    x = [float(x) for x in df0.columns.values.tolist()]
    y = x_multiplier*np.arange(section_y[0]+1, section_y[1]+1, 1)

    # Work with numeric values only; keep NaN as missing.
    z_arr = df0.to_numpy(dtype=np.float64, copy=True)
    z_arr[~np.isfinite(z_arr)] = np.nan
    finite_vals = z_arr[np.isfinite(z_arr)]

    if section_z is None:
        z_step = 0
        z_step_size = 0.0001
        z_alternator = 0
        while (z_step > 16) or (z_step == 0):
            if (z_alternator == 0):
                z_step_size = z_step_size * 2.5
            if (z_alternator == 1):
                z_step_size = z_step_size * 2
            if (z_alternator == 2):
                z_step_size = z_step_size * 2
                z_alternator = -1
            z_alternator += 1

            if finite_vals.size == 0:
                z_min, z_max = 0.0, 1.0
            else:
                z_min = float(np.nanmin(finite_vals))
                z_max = float(np.nanmax(finite_vals))

            # force minimum and maximum values
            z_min = 0

            z_min = round_nearest(z_min, z_step_size)
            z_max = round_nearest(z_max, z_step_size)+z_step_size
            z_step = int((z_max - z_min)/z_step_size)+1

        if not np.isfinite(z_min):
            z_min = 0.0
        if not np.isfinite(z_max) or z_max <= z_min:
            z_max = z_min + 1.0
        ticks = np.linspace(z_min, z_max, z_step)
        levels = np.linspace(z_min, z_max, z_colorlevels)

    else:
        z0 = float(section_z[0])
        z1 = float(section_z[1])
        if (not np.isfinite(z0)) or (not np.isfinite(z1)) or (z1 <= z0):
            if finite_vals.size == 0:
                z0, z1 = 0.0, 1.0
            else:
                z0 = 0.0
                z1 = float(np.nanmax(finite_vals))
                if (not np.isfinite(z1)) or (z1 <= z0):
                    z1 = z0 + 1.0
        ticks = np.linspace(z0, z1, int(section_z[2]))
        levels = np.linspace(z0, z1, int(section_z[3]))

    fontsize = [font_scaling*22, font_scaling*18, font_scaling*16]
    fig, ax = plt.subplots(figsize=(w, h))

    z_graylevels = np.linspace(0, 100, 101)
    z_graylevels = np.setdiff1d(z_graylevels, ticks)

    z_graylevels2 = np.linspace(0, 100, 201)
    z_graylevels2 = np.setdiff1d(z_graylevels2, ticks)
    z_graylevels2 = np.setdiff1d(z_graylevels2, z_graylevels)

    # HERE WE GO
    CS = plt.contourf(x, y, z_arr, alpha=z_alpha, cmap=cmc.batlowW_r, levels=levels)
    C = plt.contour(x, y, z_arr, z_graylevels, colors='gray',
                    linestyles="solid", linewidths=1, alpha=0.50)

    plt.clabel(C, inline=1,  inline_spacing=7, fmt="%1.1f", fontsize=fontsize[2])

    C = plt.contour(x, y, z_arr, levels=ticks, colors='black',
                    linestyles="solid", linewidths=1.5)
    plt.clabel(C, inline=1, inline_spacing=7, fmt="%1.1f", fontsize=fontsize[2])

    font = m.font_manager.FontProperties(family='Arial', style='normal', size=fontsize[2])
    fontZ = m.font_manager.FontProperties(family='Arial', style='normal', size=fontsize[2]*1.2)

    ax.yaxis.label.set_font_properties(font)
    ax.xaxis.label.set_font_properties(font)

    cbar = fig.colorbar(CS, format="%."+str(1)+"f", ticks=ticks, pad=0.03, aspect=15)
    cbar.ax.tick_params(labelsize=fontsize[1])
    cbar.set_label('$\\it{j}$ $(mA/cm^2)$', rotation=270, labelpad=30)

    ax.minorticks_on()
    ax.set_xlabel('Potential (V vs RHE)', fontsize=fontsize[0])
    ax.set_ylabel('Cycle number', fontsize=fontsize[0])

    ax.tick_params(axis='y', which='minor', bottom=True,  length=6)
    ax.tick_params(axis='y', which='major', bottom=True, width=1.3, length=7)
    ax.tick_params(axis='x', which='minor', bottom=True,  length=6)
    ax.tick_params(axis='x', which='major', bottom=True, width=1.3, length=7)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(fontZ)

    import matplotlib.ticker as ticker
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=.01))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=50))

    cbar.ax.yaxis.label.set_font_properties(font)

    plt.xticks(fontsize=fontsize[1])
    plt.yticks(fontsize=fontsize[1])

    if AxesOff:
        # Turn off tick labels
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        cbar.ax.set_yticklabels([])
        cbar.ax.set(ylabel=None)

    return fig, ax


def symlog_bins(arr, n_bins=[],  zero_eps=0.1, padding=0, base=10):

    a = np.min(arr.to_numpy().flatten().astype(np.float64))
    b = np.max(arr.to_numpy().flatten().astype(np.float64))
    a = a / (1 + padding)
    b = b * (1 + padding)
    if a > b:
        a, b = b, a

    if len(n_bins) < 3:
        z_step = 0
        z_step_size = 0.001
        z_alternator = 0
        while (z_step > 16) or (z_step == 0):
            if (z_alternator == 0):
                z_step_size = z_step_size * 2.5
            if (z_alternator == 1):
                z_step_size = z_step_size * 2
            if (z_alternator == 2):
                z_step_size = z_step_size * 2
                z_alternator = -1
            z_alternator += 1

            z_min = a
            z_max = b
            z_min = round_nearest(z_min, z_step_size)
            z_max = round_nearest(z_max, z_step_size)
            z_step = int((z_max - z_min)/z_step_size)+1
            print([z_min, z_max, z_step])
        n_bins = [z_step, z_step, z_step]

    for bix, bn in enumerate(n_bins):
        if bn < 2:
            n_bins[bix] = 2

    neg_range_log = None
    zero_range_log = None
    pos_range_log = None

    if (a < -zero_eps) and (b < -zero_eps):
        neg_range_log = [np.log10(-a)/np.log10(base), np.log10(zero_eps)/np.log10(base)]
        zero_range_log = None
        pos_range_log = None

    elif (a < -zero_eps) and (-zero_eps < b < zero_eps):
        neg_range_log = [np.log10(-a)/np.log10(base), np.log10(zero_eps)/np.log10(base)]
        zero_range_log = [-zero_eps, b]
        pos_range_log = None

    elif (zero_eps < a < -zero_eps) and (-zero_eps < b < zero_eps):
        neg_range_log = None
        zero_range_log = [a, b]
        pos_range_log = None

    elif (-zero_eps < a < zero_eps) and (b > zero_eps):
        neg_range_log = None
        zero_range_log = [a, zero_eps]
        pos_range_log = [np.log10(zero_eps)/np.log10(base), np.log10(b)]

    elif (a < -zero_eps) and (b > zero_eps):
        neg_range_log = [np.log10(-a)/np.log10(base), np.log10(zero_eps)/np.log10(base)]
        zero_range_log = [-zero_eps, zero_eps]
        pos_range_log = [np.log10(zero_eps)/np.log10(base), np.log10(b)/np.log10(base)]

    elif (a > zero_eps) and (b > zero_eps):
        neg_range_log = None
        zero_range_log = None
        pos_range_log = [np.log10(a)/np.log10(base), np.log10(b)/np.log10(base)]

    else:
        neg_range_log = None
        zero_range_log = None
        pos_range_log = None

    neg_bins = n_bins[0]
    zero_bins = n_bins[1]
    pos_bins = n_bins[2]

    neg_bin_ticks = list(-1*np.logspace(neg_range_log[0], neg_range_log[1],
                                        neg_bins, base=base))[:-1] if neg_range_log is not None else []

    zero_bin_ticks = list(np.linspace(zero_range_log[0], zero_range_log[1],
                                      zero_bins)) if zero_range_log is not None else []

    pos_bin_ticks = list(np.logspace(pos_range_log[0], pos_range_log[1], pos_bins, base=base))[  # noqa
        1:] if pos_range_log is not None else []

    result = neg_bin_ticks + zero_bin_ticks + pos_bin_ticks
    return result


def closest_round(val):
    print(val)

    z_step_size = 0.0000001*np.sign(val)
    z_alternator = 0

    while (np.sign(val)*z_step_size < np.sign(val)*val):
        if (z_alternator == 0):
            z_step_size = z_step_size * 2
        if (z_alternator == 1):
            z_step_size = z_step_size/2 * 3
        if (z_alternator == 2):
            z_step_size = z_step_size/3 * 4
        if (z_alternator == 3):
            z_step_size = z_step_size/4 * 5
        if (z_alternator == 4):
            z_step_size = z_step_size/5 * 6
        if (z_alternator == 5):
            z_step_size = z_step_size/6 * 7
        if (z_alternator == 6):
            z_step_size = z_step_size/7 * 8
        if (z_alternator == 7):
            z_step_size = z_step_size/8 * 9
        if (z_alternator == 8):
            z_step_size = z_step_size/9 * 10
            z_alternator = -1
        z_alternator += 1

    return z_step_size


def removeYinterval(df, r0, r1):
    """
    Interpolate over a vertical block of rows [r0:r1) (0-based indexing).
    """
    p0 = df.iloc[r0-2:r0-1, :].copy()
    p1 = df.iloc[r1+0:r1+1, :].copy()

    for n in range(len(p0.columns)):
        intp = np.linspace(p0.iat[0, n],  p1.iat[0, n], r1-r0+2)
        for mm in range(r0, r1, 1):
            df.iat[mm, n] = intp[mm-r0+1]

    return df


def suppress_every_nth_cycle(df, start_cycle=1, every_n=0, mode="nan"):
    """
    Suppress every Nth cycle row starting at cycle M (1-based row index).

    mode:
      - "nan": keep rows but set values to NaN
      - "drop": remove rows entirely
      - "interp": replace suppressed rows by interpolation over cycle index
    """
    if every_n is None or int(every_n) <= 0:
        return df, 0

    out = df.copy()
    n_rows = len(out)
    if n_rows == 0:
        return out, 0

    start_idx = max(int(start_cycle) - 1, 0)
    row_idx = np.arange(n_rows)
    mask = (row_idx >= start_idx) & (((row_idx - start_idx) % int(every_n)) == 0)
    n_suppressed = int(mask.sum())

    if mode == "drop":
        out = out.loc[~mask].copy()
    elif mode == "interp":
        # Remove selected rows from the data contribution, then reconstruct
        # them from neighboring cycles to avoid visible white gaps in contour.
        out.iloc[mask, :] = np.nan
        out = out.interpolate(axis=0, method="linear", limit_direction="both")
    else:
        out.iloc[mask, :] = np.nan

    return out, n_suppressed


def smooth_contour_matrix(df, sigma_cycle=1.0, sigma_potential=1.0):
    """
    NaN-aware 2D Gaussian smoothing.
    sigma_cycle: smoothing along row/cycle direction
    sigma_potential: smoothing along potential/column direction
    """
    z = df.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(z)
    if not finite.any():
        return df.copy()

    values = np.where(finite, z, 0.0)
    weights = finite.astype(np.float64)

    sigma = (float(sigma_cycle), float(sigma_potential))
    values_s = gaussian_filter(values, sigma=sigma, mode="nearest")
    weights_s = gaussian_filter(weights, sigma=sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        z_s = values_s / weights_s
    z_s[weights_s <= 1e-12] = np.nan

    return pd.DataFrame(z_s, index=df.index.copy(), columns=df.columns.copy())


def fit_double_line(E_ox, j_ox, E_red, j_red, weight_ox=2.0, weight_red=1.0):
    """
    Fit two parallel lines (common slope, separate intercepts)
    to ox and red segments using weighted least squares.

    Model:
        For ox points:  j = m * E + b_ox
        For red points: j = m * E + b_red

    Returns:
        m, b_ox, b_red
    """
    E_ox = np.asarray(E_ox, dtype=float)
    E_red = np.asarray(E_red, dtype=float)
    j_ox = np.asarray(j_ox, dtype=float)
    j_red = np.asarray(j_red, dtype=float)

    n_ox = len(E_ox)
    n_red = len(E_red)

    x_all = np.concatenate([E_ox, E_red])
    y_all = np.concatenate([j_ox, j_red])

    I_ox = np.concatenate([np.ones(n_ox), np.zeros(n_red)])
    I_red = np.concatenate([np.zeros(n_ox), np.ones(n_red)])

    X = np.column_stack([x_all, I_ox, I_red])

    w = np.concatenate([
        np.full(n_ox, weight_ox, dtype=float),
        np.full(n_red, weight_red, dtype=float),
    ])
    W_sqrt = np.sqrt(w)

    Xw = X * W_sqrt[:, None]
    yw = y_all * W_sqrt

    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    m_fit, b_ox_fit, b_red_fit = beta
    return m_fit, b_ox_fit, b_red_fit


# =====================================================================================
# Per-directory processing (uses JSON only)
# =====================================================================================

def process_out_dir(process_dir):
    """
    process_dir: path to a directory that contains:
        df_ox.json
        df_red.json
    Outputs:
        contour plots, hysteresis CSV+plots, double-line fit plots,
        10 representative CV+fit plots, single-cycle CV plots
        saved into process_dir.
    """
    print(f"\nProcessing directory: {process_dir}")

    # Figure out a nicer base name for files (strip "out_" prefix if present)
    base_dirname = os.path.basename(process_dir.rstrip(os.sep))
    if base_dirname.startswith("out_"):
        exp_filename = base_dirname[4:]
    else:
        exp_filename = base_dirname

    # -------------------------------------------------------------------------
    # Load matrices
    # -------------------------------------------------------------------------
    ox_path = os.path.join(process_dir, "df_ox.json")
    red_path = os.path.join(process_dir, "df_red.json")

    if not (os.path.isfile(ox_path) and os.path.isfile(red_path)):
        print("  -> Missing df_ox.json or df_red.json; skipping this directory.")
        return

    df_cvloop_ox = pd.read_json(ox_path, convert_axes=False)
    df_cvloop_red = pd.read_json(red_path, convert_axes=False)

    # -------------------------------------------------------------------------
    # Contour plot (same processing as before)
    # -------------------------------------------------------------------------
    print("  [1/4] Making contour plots from df_ox.json ...")

    df_ox_contour = df_cvloop_ox.copy()

    # Convert to current density
    df_ox_contour /= 0.196

    # Shift potentials to V vs RHE
    xa = df_ox_contour.columns.values.tolist()
    x = [str(float(xx) + 0.000) for xx in xa]
    df_ox_contour.columns = x

    # S1 patching (same hard-coded ranges as your original)
    # df_ox_contour = removeYinterval(df_ox_contour, 350, 400)
    # df_ox_contour = removeYinterval(df_ox_contour, 235, 290)
    x_multiplier = 5

    # Downsample in cycle-direction
    df_ox_contour = df_ox_contour[::x_multiplier]

    # Optional suppression of contour rows:
    # suppress every Nth displayed cycle row starting from cycle M.
    # Set every_n <= 0 to disable.
    suppress_start_cycle = 1
    suppress_every_n = 20
    suppress_mode = "interp"   # "nan", "interp", or "drop"
    df_ox_contour, n_suppressed = suppress_every_nth_cycle(
        df_ox_contour,
        start_cycle=suppress_start_cycle,
        every_n=suppress_every_n,
        mode=suppress_mode,
    )
    if n_suppressed > 0:
        print(f"      Suppressed {n_suppressed} contour row(s): "
              f"every {suppress_every_n}th from cycle {suppress_start_cycle} (mode={suppress_mode})")

    # Optional 2D smoothing for contour visualization.
    smooth_contour = True
    smooth_sigma_cycle = 0.8       # rows (cycle direction)
    smooth_sigma_potential = 1.0   # columns (potential direction)
    if smooth_contour:
        df_ox_contour = smooth_contour_matrix(
            df_ox_contour,
            sigma_cycle=smooth_sigma_cycle,
            sigma_potential=smooth_sigma_potential,
        )
        print("      Applied contour smoothing: "
              f"sigma_cycle={smooth_sigma_cycle}, sigma_potential={smooth_sigma_potential}")

    df0_x = [float(xx) for xx in df_ox_contour.columns]
    section_x = [min(df0_x), max(df0_x)]
    section_y = [1, len(df_ox_contour)]
    z_vals = df_ox_contour.to_numpy(dtype=np.float64, copy=True)
    z_vals = z_vals[np.isfinite(z_vals)]
    z_setMaximum = float(np.max(z_vals)) if z_vals.size else 1.0
    if z_setMaximum <= 0:
        z_setMaximum = 1.0

    # Shift the lower bound below zero so 0 is not mapped to the first color.
    # Increase this fraction if you want 0 to sit further into the colormap.
    z_zero_shift_frac = 0.08
    z_setMinimum = -z_zero_shift_frac * z_setMaximum
    section_z = [z_setMinimum, z_setMaximum, 11, 256]

    z_sublevels = 12
    z_colorlevels = 256
    z_alpha = 1

    # Contour with axes
    fig, ax = plotContour(df_ox_contour,
                          x_multiplier=x_multiplier,
                          section_x=section_x,
                          section_y=section_y,
                          section_z=section_z,
                          z_sublevels=z_sublevels,
                          z_colorlevels=z_colorlevels,
                          z_alpha=z_alpha,
                          AxesOff=False)
    filename4 = os.path.join(process_dir, make_safe_filename(exp_filename))
    ensure_dir(filename4)
    fig.savefig(filename4 + "150dpi", dpi=150, bbox_inches="tight")
    fig.savefig(filename4 + "300dpi", dpi=300, bbox_inches="tight")
    fig.savefig(filename4 + "600dpi", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Contour without axes
    fig, ax = plotContour(df_ox_contour,
                          x_multiplier=x_multiplier,
                          section_x=section_x,
                          section_y=section_y,
                          section_z=section_z,
                          z_sublevels=z_sublevels,
                          z_colorlevels=z_colorlevels,
                          z_alpha=z_alpha,
                          AxesOff=True)
    fig.savefig(filename4 + "150dpi_noAxes", dpi=150, bbox_inches="tight")
    fig.savefig(filename4 + "300dpi_noAxes", dpi=300, bbox_inches="tight")
    fig.savefig(filename4 + "600dpi_noAxes", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # [NEW-ish] Hysteresis vs cycle at fixed potentials (missing-cycle-proof)
    # -------------------------------------------------------------------------
    print("  [2/4] Computing hysteresis vs cycle ...")

    # ======= USER TUNABLE PART =======
    # Potentials in V vs RHE where you want to sample j:
    target_potential_RHE_ox = 1.1   # forward / ox branch
    target_potential_RHE_red = 1.1  # backward / red branch
    potential_offset = 0       # JSON potential + offset = V vs RHE
    # =================================

    # Reload original (unscaled) matrices
    df_ox_raw = pd.read_json(ox_path, convert_axes=False)
    df_red_raw = pd.read_json(red_path, convert_axes=False)

    # Column potentials in "JSON scale"
    E_ox = np.array([float(e) for e in df_ox_raw.columns])
    E_red = np.array([float(e) for e in df_red_raw.columns])

    # Convert RHE target to JSON scale
    target_potential_json_ox = target_potential_RHE_ox - potential_offset
    target_potential_json_red = target_potential_RHE_red - potential_offset

    # Nearest column indices
    idx_ox = (np.abs(E_ox - target_potential_json_ox)).argmin()
    idx_red = (np.abs(E_red - target_potential_json_red)).argmin()

    E_ox_sel_json = E_ox[idx_ox]
    E_red_sel_json = E_red[idx_red]

    E_ox_sel_RHE = E_ox_sel_json + potential_offset
    E_red_sel_RHE = E_red_sel_json + potential_offset

    print(f"      Forward scan: target {target_potential_RHE_ox:.4f} V vs RHE "
          f"-> using {E_ox_sel_RHE:.4f} V vs RHE (col {idx_ox})")
    print(f"      Backward scan: target {target_potential_RHE_red:.4f} V vs RHE "
          f"-> using {E_red_sel_RHE:.4f} V vs RHE (col {idx_red})")

    # Extract j(E) at those potentials, convert to current density (mA/cm^2)
    s_ox = df_ox_raw.iloc[:, idx_ox].astype(float) / 0.196
    s_red = df_red_raw.iloc[:, idx_red].astype(float) / 0.196

    s_ox.name = "j_forward_mAcm2"
    s_red.name = "j_backward_mAcm2"

    # --- build explicit cycle numbers from index if possible ---
    def extract_cycle_index(idx):
        # try to convert index labels to numeric cycles
        cyc = pd.to_numeric(idx, errors="coerce")
        if cyc.isna().all():
            # total failure → use position index 1..N
            return np.arange(1, len(idx) + 1, dtype=int)
        return cyc.to_numpy()

    cycles_ox = extract_cycle_index(df_ox_raw.index)
    cycles_red = extract_cycle_index(df_red_raw.index)

    df_ox_c = pd.DataFrame({
        "cycle": cycles_ox,
        "j_forward_mAcm2": s_ox.values,
    })

    df_red_c = pd.DataFrame({
        "cycle": cycles_red,
        "j_backward_mAcm2": s_red.values,
    })

    # inner join on cycle number
    df_hys = pd.merge(df_ox_c, df_red_c, on="cycle", how="inner")

    print(f"      cycles in ox:  {len(df_ox_c)}")
    print(f"      cycles in red: {len(df_red_c)}")
    print(f"      cycles in common (by label): {len(df_hys)}")

    # Fallback: if there is no overlap at all, align by position
    if df_hys.empty:
        print("      No overlapping cycle labels; falling back to position-based alignment.")
        n_common = min(len(s_ox), len(s_red))
        df_hys = pd.DataFrame({
            "cycle": np.arange(1, n_common + 1, dtype=int),
            "j_forward_mAcm2": s_ox.iloc[:n_common].values,
            "j_backward_mAcm2": s_red.iloc[:n_common].values,
        })

    # Δj = j_forward - j_backward
    df_hys["delta_j_mAcm2"] = df_hys["j_forward_mAcm2"] - df_hys["j_backward_mAcm2"]

    cycles = df_hys["cycle"].to_numpy()

    outdir = process_dir
    safe_E = make_safe_filename(
        f"ox_{target_potential_RHE_ox:.3f}V__red_{target_potential_RHE_red:.3f}V"
    )

    # Save CSV
    hys_csv = os.path.join(outdir, f"hysteresis_{safe_E}.csv")
    ensure_dir(hys_csv)
    df_hys.to_csv(hys_csv, index=False)

    # Plot j_ox, j_red, and Δj vs cycle
    fig_hys, ax_hys = plt.subplots(figsize=(8, 5))
    ax_hys.plot(cycles, df_hys["j_forward_mAcm2"].to_numpy(),
                marker="o", linewidth=1.5, label="j_ox (forward)")
    ax_hys.plot(cycles, df_hys["j_backward_mAcm2"].to_numpy(),
                marker="o", linewidth=1.5, label="j_red (backward)")
    ax_hys.plot(cycles, df_hys["delta_j_mAcm2"].to_numpy(),
                marker="o", linewidth=1.5, label="Δj = j_ox − j_red")

    ax_hys.axhline(0, color="black", linewidth=1)

    ax_hys.set_xlabel("Cycle number")
    ax_hys.set_ylabel(r"j / Δj (mA/cm$^2$)")
    ax_hys.set_title(
        "Forward–Backward currents and difference\n"
        f"ox @ {E_ox_sel_RHE:.3f} V vs RHE, "
        f"red @ {E_red_sel_RHE:.3f} V vs RHE"
    )
    ax_hys.legend()
    fig_hys.tight_layout()

    fig_hys.savefig(os.path.join(outdir, f"hysteresis_{safe_E}_150dpi.png"),
                    dpi=150, bbox_inches="tight")
    fig_hys.savefig(os.path.join(outdir, f"hysteresis_{safe_E}_300dpi.png"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig_hys)

    # -------------------------------------------------------------------------
    # [NEW] Double-line fit: parallel ox/red segments per cycle
    # -------------------------------------------------------------------------
    print("  [3/4] Computing double-line offset vs cycle ...")

    # User tunables for the double-line fit
    ox_window_RHE = (1.0, 1.05)     # V vs RHE, oxidation branch window
    red_window_RHE = (1.0, 1.05)   # V vs RHE, reduction branch window
    weight_ox = 1.0                # relative weight of ox points
    weight_red = 1.0               # relative weight of red points

    # Potentials in V vs RHE for all columns
    E_ox_RHE = E_ox + potential_offset
    E_red_RHE = E_red + potential_offset

    # Column selection for the windows (same columns for all cycles)
    mask_ox = (E_ox_RHE >= ox_window_RHE[0]) & (E_ox_RHE <= ox_window_RHE[1])
    mask_red = (E_red_RHE >= red_window_RHE[0]) & (E_red_RHE <= red_window_RHE[1])

    if mask_ox.sum() < 2 or mask_red.sum() < 2:
        print("      Not enough points in ox/red windows to perform double-line fit; skipping.")
        df_double = None
    else:
        n_common = min(df_ox_raw.shape[0], df_red_raw.shape[0])

        cycles_used = []
        delta_offsets = []
        m_fits = []
        b_ox_fits = []
        b_red_fits = []

        for i in range(n_common):
            # current density for this cycle
            j_ox_cycle = np.asarray(df_ox_raw.iloc[i, :], dtype=float) / 0.196
            j_red_cycle = np.asarray(df_red_raw.iloc[i, :], dtype=float) / 0.196

            y_ox_seg = j_ox_cycle[mask_ox]
            y_red_seg = j_red_cycle[mask_red]

            # Remove NaNs
            valid_ox = np.isfinite(y_ox_seg)
            valid_red = np.isfinite(y_red_seg)

            if valid_ox.sum() < 2 or valid_red.sum() < 2:
                continue

            E_ox_seg = E_ox_RHE[mask_ox][valid_ox]
            E_red_seg = E_red_RHE[mask_red][valid_red]
            y_ox_seg = y_ox_seg[valid_ox]
            y_red_seg = y_red_seg[valid_red]

            try:
                m_fit, b_ox_fit, b_red_fit = fit_double_line(
                    E_ox_seg, y_ox_seg,
                    E_red_seg, y_red_seg,
                    weight_ox=weight_ox,
                    weight_red=weight_red
                )
            except np.linalg.LinAlgError:
                continue

            # Cycle label from df_ox_c (position-based mapping)
            if i < len(df_ox_c):
                cycle_label = df_ox_c.iloc[i]["cycle"]
            else:
                cycle_label = i + 1

            cycles_used.append(int(cycle_label))
            # Vertical difference between parallel lines (constant over x):
            delta_offsets.append(b_ox_fit - b_red_fit)
            m_fits.append(m_fit)
            b_ox_fits.append(b_ox_fit)
            b_red_fits.append(b_red_fit)

        if len(cycles_used) == 0:
            print("      No cycles could be fitted for double-line offset.")
            df_double = None
        else:
            df_double = pd.DataFrame({
                "cycle": cycles_used,
                "delta_line_mAcm2": delta_offsets,
                "m_fit": m_fits,
                "b_ox_fit": b_ox_fits,
                "b_red_fit": b_red_fits,
            })

            # Outlier removal for delta_line_mAcm2 (z-score) on sampled points
            plot_every_n = 10
            plot_start_index = 8

            delta_vals = df_double["delta_line_mAcm2"].to_numpy()
            idx_sample = np.arange(plot_start_index, len(delta_vals), plot_every_n)
            delta_sample = delta_vals[idx_sample]

            # z-score method
            outlier_zscore_thresh = 50
            mean = np.nanmean(delta_sample)
            std = np.nanstd(delta_sample, ddof=0)
            if std == 0 or not np.isfinite(std):
                is_outlier_z_sample = np.zeros_like(delta_sample, dtype=bool)
            else:
                z = (delta_sample - mean) / std
                is_outlier_z_sample = np.abs(z) > outlier_zscore_thresh

            is_outlier_z = np.full(len(delta_vals), np.nan)
            is_outlier_z[idx_sample] = is_outlier_z_sample

            delta_no_z = np.full(len(delta_vals), np.nan)
            delta_no_z[idx_sample] = np.where(is_outlier_z_sample, np.nan, delta_sample)

            df_double["is_outlier_zscore"] = is_outlier_z
            df_double["delta_line_mAcm2_no_outliers_zscore"] = delta_no_z
            df_double["is_sampled"] = False
            df_double.loc[idx_sample, "is_sampled"] = True

            # Save CSV
            dl_csv = os.path.join(outdir, "double_line_offset.csv")
            ensure_dir(dl_csv)
            df_double.to_csv(dl_csv, index=False)

            # Plot Δy between the two parallel lines vs cycle
            fig_dl, ax_dl = plt.subplots(figsize=(8, 5))
            cycles_arr = df_double["cycle"].to_numpy()

            idx_plot = idx_sample

            ax_dl.plot(cycles_arr[idx_plot],
                       df_double["delta_line_mAcm2_no_outliers_zscore"].to_numpy()[idx_plot],
                       marker="o", linewidth=1.5, label="z-score (no outliers)")

            ax_dl.axhline(0, color="black", linewidth=1)
            ax_dl.set_xlabel("Cycle number")
            ax_dl.set_ylabel(r"Δy between parallel fits (mA/cm$^2$)")
            ax_dl.set_title(
                "Double-line fit offset vs cycle\n"
                f"ox window [{ox_window_RHE[0]:.2f}, {ox_window_RHE[1]:.2f}] V, "
                f"red window [{red_window_RHE[0]:.2f}, {red_window_RHE[1]:.2f}] V\n"
                f"weights: ox={weight_ox}, red={weight_red}"
            )
            ax_dl.legend()
            fig_dl.tight_layout()

            base = os.path.join(outdir, "double_line_offset")
            fig_dl.savefig(base + "_150dpi.png",
                           dpi=150, bbox_inches="tight")
            fig_dl.savefig(base + "_300dpi.png",
                           dpi=300, bbox_inches="tight")
            plt.close(fig_dl)

            # -----------------------------------------------------------------
            # Generate up to 10 representative CV plots with fitted lines
            # -----------------------------------------------------------------
            n_plots = min(10, len(df_double))
            if n_plots > 0:
                print(f"      Generating {n_plots} CV+fit plots (equally spaced in fitted cycles) ...")
                # indices equally spaced across df_double rows
                idxs = np.linspace(0, len(df_double) - 1, n_plots)
                idxs = np.unique(np.round(idxs).astype(int))

                for idx in idxs:
                    row = df_double.iloc[idx]
                    cyc = int(row["cycle"])
                    m_fit = row["m_fit"]
                    b_ox_fit = row["b_ox_fit"]
                    b_red_fit = row["b_red_fit"]
                    delta_y = row["delta_line_mAcm2"]

                    # Determine row index in df_ox_raw / df_red_raw corresponding to this cycle
                    row_idx = None
                    if np.issubdtype(df_ox_raw.index.dtype, np.number):
                        idx_candidates = np.where(df_ox_raw.index.to_numpy() == cyc)[0]
                        if len(idx_candidates) > 0:
                            row_idx = int(idx_candidates[0])
                    if row_idx is None:
                        # fall back to df_ox_c mapping
                        idx_candidates = np.where(df_ox_c["cycle"].to_numpy() == cyc)[0]
                        if len(idx_candidates) > 0:
                            row_idx = int(idx_candidates[0])

                    if row_idx is None or row_idx >= df_ox_raw.shape[0] or row_idx >= df_red_raw.shape[0]:
                        continue

                    j_ox_cycle = np.asarray(df_ox_raw.iloc[row_idx, :], dtype=float) / 0.196
                    j_red_cycle = np.asarray(df_red_raw.iloc[row_idx, :], dtype=float) / 0.196

                    fig_cvfit, ax_cvfit = plt.subplots(figsize=(8, 5))

                    # Full CV curves
                    ax_cvfit.plot(E_ox_RHE, j_ox_cycle, label="Forward (ox)", linewidth=2)
                    ax_cvfit.plot(E_red_RHE, j_red_cycle, label="Backward (red)", linewidth=2)

                    # Highlight segments used for fit
                    ax_cvfit.plot(E_ox_RHE[mask_ox], j_ox_cycle[mask_ox],
                                  linestyle="", marker="o", markersize=4,
                                  label="Ox fit window")
                    ax_cvfit.plot(E_red_RHE[mask_red], j_red_cycle[mask_red],
                                  linestyle="", marker="s", markersize=4,
                                  label="Red fit window")

                    # Fitted lines extrapolated by 50% of window width on both sides
                    E_ox_fit = E_ox_RHE[mask_ox]
                    E_red_fit = E_red_RHE[mask_red]
                    ox_range = E_ox_fit.max() - E_ox_fit.min()
                    red_range = E_red_fit.max() - E_red_fit.min()
                    E_ox_ext = np.linspace(E_ox_fit.min() - 0.5 * ox_range,
                                           E_ox_fit.max() + 0.5 * ox_range, 100)
                    E_red_ext = np.linspace(E_red_fit.min() - 0.5 * red_range,
                                            E_red_fit.max() + 0.5 * red_range, 100)
                    j_ox_ext = m_fit * E_ox_ext + b_ox_fit
                    j_red_ext = m_fit * E_red_ext + b_red_fit

                    ax_cvfit.plot(E_ox_ext, j_ox_ext, linestyle="-", linewidth=1,
                                  label="Ox fit (extrapolated)")
                    ax_cvfit.plot(E_red_ext, j_red_ext, linestyle="-", linewidth=1,
                                  label="Red fit (extrapolated)")

                    ax_cvfit.axhline(0, color="black", linewidth=1)

                    ax_cvfit.set_xlabel("Potential (V vs RHE)")
                    ax_cvfit.set_ylabel(r"$j$ (mA/cm$^2$)")
                    ax_cvfit.set_title(f"CV with double-line fit, cycle {cyc}")

                    # Small annotation with Δy
                    textstr = (f"cycle = {cyc}\n"
                               f"Δy (b_ox - b_red) = {delta_y:.3f} mA/cm²")
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.25)
                    ax_cvfit.text(0.02, 0.98, textstr, transform=ax_cvfit.transAxes,
                                  fontsize=10, verticalalignment='top',
                                  horizontalalignment='left', bbox=props)

                    ax_cvfit.legend()
                    fig_cvfit.tight_layout()

                    base_cvfit = os.path.join(outdir, f"CV_fit_cycle{cyc}")
                    fig_cvfit.savefig(base_cvfit + "_150dpi.png",
                                      dpi=150, bbox_inches="tight")
                    fig_cvfit.savefig(base_cvfit + "_300dpi.png",
                                      dpi=300, bbox_inches="tight")
                    plt.close(fig_cvfit)

    # -------------------------------------------------------------------------
    # Single CV for a given cycle (forward + backward)
    # -------------------------------------------------------------------------
    print("  [4/4] Plotting single CV (forward + backward) ...")

    target_cycle_number = 20  # logical cycle you care about

    # Determine row index for that global cycle index
    if np.issubdtype(df_ox_raw.index.dtype, np.number):
        idx_candidates = np.where(df_ox_raw.index.to_numpy() == target_cycle_number)[0]
        if len(idx_candidates) == 0:
            print(f"      Requested global_cycle {target_cycle_number} not found; "
                  f"using row index 19 (if available).")
            row_idx = min(19, df_ox_raw.shape[0]-1)
            cycle_label = df_ox_raw.index[row_idx]
        else:
            row_idx = int(idx_candidates[0])
            cycle_label = target_cycle_number
    else:
        print("      Non-numeric index; using row index 19 (if available).")
        row_idx = min(19, df_ox_raw.shape[0]-1)
        cycle_label = row_idx + 1

    j_ox_cycle = np.asarray(df_ox_raw.iloc[row_idx, :], dtype=float) / 0.196
    j_red_cycle = np.asarray(df_red_raw.iloc[row_idx, :], dtype=float) / 0.196

    E_ox_RHE = E_ox + potential_offset
    E_red_RHE = E_red + potential_offset

    fig_cv, ax_cv = plt.subplots(figsize=(8, 5))
    ax_cv.plot(E_ox_RHE, j_ox_cycle, label="Forward (ox)", linewidth=2)
    ax_cv.plot(E_red_RHE, j_red_cycle, label="Backward (red)", linewidth=2)

    ax_cv.axhline(0, color="black", linewidth=1)
    ax_cv.set_xlabel("Potential (V vs RHE)")
    ax_cv.set_ylabel(r"$j$ (mA/cm$^2$)")
    ax_cv.set_title(f"CV, cycle {cycle_label}")
    ax_cv.legend()
    fig_cv.tight_layout()

    cv_base = os.path.join(outdir, f"CV_cycle{cycle_label}")
    fig_cv.savefig(cv_base + "_150dpi.png",
                   dpi=150, bbox_inches="tight")
    fig_cv.savefig(cv_base + "_300dpi.png",
                   dpi=300, bbox_inches="tight")
    plt.close(fig_cv)

    print(f"  Done with {process_dir}")
    return df_double, exp_filename


# =====================================================================================
# Main: scan for out_* dirs with JSON and process them
# =====================================================================================

if __name__ == "__main__":
    rootdir = "./"
    dirListToProcess = []

    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            if file.startswith("_"):
                continue
            ox_here = os.path.isfile(os.path.join(d, "df_ox.json"))
            red_here = os.path.isfile(os.path.join(d, "df_red.json"))
            if ox_here and red_here:
                dirListToProcess.append(d)

    print("Will process the following JSON directories:")
    for d in dirListToProcess:
        print("  ", d)

    all_double = []
    for process_dir in dirListToProcess:
        df_double, exp_name = process_out_dir(process_dir)
        if df_double is not None and len(df_double) > 0:
            df_copy = df_double.copy()
            df_copy["sample"] = exp_name
            all_double.append(df_copy)

    if len(all_double) > 0:
        keep_cols = [
            "cycle",
            "delta_line_mAcm2",
            "m_fit",
            "b_ox_fit",
            "b_red_fit",
            "is_outlier_zscore",
            "delta_line_mAcm2_no_outliers_zscore",
            "is_sampled",
            "sample",
        ]
        df_all = pd.concat(all_double, ignore_index=True)[keep_cols]
        df_all["cycle"] = (df_all["cycle"].astype(float) + 1) / 2

        frames = []
        for sample, df_s in df_all.groupby("sample"):
            df_s = df_s.drop(columns=["sample"]).copy()
            df_s = df_s.sort_values("cycle")
            df_s = df_s.set_index("cycle")
            df_s.columns = pd.MultiIndex.from_product([df_s.columns, [sample]])
            frames.append(df_s)

        df_wide = pd.concat(frames, axis=1)
        df_wide = df_wide.sort_index()

        out_csv = os.path.join(rootdir, "double_line_offset_all_samples.csv")
        ensure_dir(out_csv)
        df_out = df_wide.copy()
        df_out.insert(0, ("cycle", ""), df_out.index.to_numpy())
        df_out = df_out.reset_index(drop=True)
        df_out.to_csv(out_csv, index=False)
        print(f"Saved combined double_line_offset CSV: {out_csv}")

        df_sampled = df_all[df_all["is_sampled"]].copy()
        frames_sampled = []
        for sample, df_s in df_sampled.groupby("sample"):
            df_s = df_s.drop(columns=["sample", "is_sampled"]).copy()
            df_s = df_s.sort_values("cycle")
            df_s = df_s.set_index("cycle")
            df_s.columns = pd.MultiIndex.from_product([df_s.columns, [sample]])
            frames_sampled.append(df_s)

        if len(frames_sampled) > 0:
            df_wide_sampled = pd.concat(frames_sampled, axis=1)
            df_wide_sampled = df_wide_sampled.sort_index()
            out_csv_sampled = os.path.join(rootdir, "double_line_offset_all_samples_sampled.csv")
            ensure_dir(out_csv_sampled)
            df_out_sampled = df_wide_sampled.copy()
            df_out_sampled.insert(0, ("cycle", ""), df_out_sampled.index.to_numpy())
            df_out_sampled = df_out_sampled.reset_index(drop=True)
            df_out_sampled.to_csv(out_csv_sampled, index=False)
            print(f"Saved sampled double_line_offset CSV: {out_csv_sampled}")
