#!/usr/bin/env python
"""
Import EC-Lab-style TXT CV files, use a Schmitt-trigger-like detector
on control/V to define full oxidation/reduction sweeps, build interpolated
matrices, compute per-sweep stats, and save JSON/CSV.

Outputs per TXT file:
  out_<safe_txt_name>/
    df_red.json       # reduction sweeps vs potential, rows = global_cycle
    df_ox.json        # oxidation sweeps vs potential, rows = global_cycle
    cycle_stats.json  # per-sweep stats (charges, ranges, etc.)
    df_red.csv
    df_ox.csv
    cycle_stats.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import interpolate

# =====================================================================================
# Settings
# =====================================================================================

force_reprocessing = True   # set True to overwrite existing JSON

# interpolation grid settings
EXTRACT_X_STEPSIZE = 0.005   # 5 mV grid
X_COL_DEFAULT = "control/V"  # x-axis for interpolation
Y_COL_DEFAULT = "<I>/mA"     # y-axis for interpolation
TIME_COL_DEFAULT = "time/s"
CTRL_COL_DEFAULT = "control/V"
CYCLE_COL_DEFAULT = "cycle number"

# minimum number of valid datapoints (per row) to accept a cycle in the matrices
MIN_POINTS_PER_ROW = 5

# threshold for "real" dE vs noise: slope_eps = SLOPE_FRAC * median(|ΔE|)
SLOPE_FRAC = 0.2

# Schmitt trigger thresholds (for a 0.0–0.6 V window)
HIGH_CUTOFF = 0.59   # V, upper threshold near 0.6 V
LOW_CUTOFF = 0.01   # V, lower threshold near 0.0 V

# Monotonicity requirement: at least this fraction of non-zero dE
# must have the same sign (rising or falling).
MONOTONICITY_FRACTION = 0.98


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
    """
    Strip a filename to only alphanumeric + underscores.
    """
    def safe_char(c):
        return c if c.isalnum() else "_"
    return "".join(safe_char(c) for c in s).rstrip("_")


def read_cv_txt(path):
    """
    Read an EC-Lab style TXT file (tab-separated) into a DataFrame.
    Assumes the first line is a header.
    """
    print(f"  [1/5] Reading TXT into DataFrame: {path}")
    df = pd.read_csv(path, sep="\t", engine="c")
    df.columns = [c.strip() for c in df.columns]

    # Drop accidental empty trailing columns from tab-terminated headers/rows.
    df = df.loc[:, [c for c in df.columns if c != ""]]

    # Normalize potential column naming across EC-Lab export variants.
    if CTRL_COL_DEFAULT not in df.columns and "Ewe/V" in df.columns:
        df[CTRL_COL_DEFAULT] = df["Ewe/V"]
        print("      -> mapped Ewe/V -> control/V")

    # Ensure cycle column exists even in minimal exports.
    if CYCLE_COL_DEFAULT not in df.columns:
        df[CYCLE_COL_DEFAULT] = 1
        print("      -> created missing cycle number column (default=1)")

    # Normalize time column to numeric seconds.
    if TIME_COL_DEFAULT not in df.columns:
        df[TIME_COL_DEFAULT] = np.arange(len(df), dtype=float)
        print("      -> created missing time/s from row index")
    else:
        t_num = pd.to_numeric(df[TIME_COL_DEFAULT], errors="coerce")
        if t_num.notna().sum() < max(2, int(0.5 * len(df))):
            t_dt = pd.to_datetime(df[TIME_COL_DEFAULT], errors="coerce")
            if t_dt.notna().sum() >= max(2, int(0.5 * len(df))):
                t0 = t_dt[t_dt.notna()].iloc[0]
                df[TIME_COL_DEFAULT] = (t_dt - t0).dt.total_seconds()
                print("      -> converted timestamp time/s to elapsed seconds")
            else:
                df[TIME_COL_DEFAULT] = np.arange(len(df), dtype=float)
                print("      -> could not parse time/s; using row index seconds")
        else:
            df[TIME_COL_DEFAULT] = t_num
    print(f"      -> shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# =====================================================================================
# Schmitt-trigger-based segment detection
# =====================================================================================

def _detect_schmitt_segments_in_session(df_sess,
                                        ctrl_col=CTRL_COL_DEFAULT,
                                        high_cutoff=HIGH_CUTOFF,
                                        low_cutoff=LOW_CUTOFF,
                                        monotonicity_fraction=MONOTONICITY_FRACTION):
    """
    Detect monotonic sweeps in a single time-ordered session using a
    Schmitt-trigger style criterion:

      * A reduction segment starts when E >= high_cutoff and dE < 0
        and ends once E <= low_cutoff.
      * An oxidation segment starts when E <= low_cutoff and dE > 0
        and ends once E >= high_cutoff.
      * A segment is accepted only if at least `monotonicity_fraction`
        of its non-zero dE steps have the correct sign (negative for
        reduction, positive for oxidation), ignoring tiny |dE| <
        slope_eps (noise).

    Returns a list of segments:
      [{"kind": "reduction" or "oxidation",
        "start_idx": int(df index),
        "end_idx": int(df index)}]
    """
    idx_all = df_sess.index.to_numpy()
    E = df_sess[ctrl_col].to_numpy(dtype=float)

    segments = []
    if len(E) < 2:
        return segments

    dE_all = np.diff(E)
    step_nonzero = np.abs(dE_all)
    step_nonzero = step_nonzero[step_nonzero > 0]

    if len(step_nonzero) == 0:
        slope_eps = 0.0
    else:
        slope_eps = SLOPE_FRAC * float(np.median(step_nonzero))

    state = "IDLE"   # "IDLE", "IN_RED", "IN_OX"
    start_pos = None

    for pos in range(len(E) - 1):
        e0 = E[pos]
        e1 = E[pos + 1]

        if np.isnan(e0) or np.isnan(e1):
            # break any ongoing segment on NaNs
            if state != "IDLE":
                state = "IDLE"
                start_pos = None
            continue

        de = e1 - e0

        if state == "IDLE":
            # look for a clean start near the upper or lower threshold
            if e0 >= high_cutoff and de < 0:
                # start of a reduction sweep (going down from high)
                state = "IN_RED"
                start_pos = pos
            elif e0 <= low_cutoff and de > 0:
                # start of an oxidation sweep (going up from low)
                state = "IN_OX"
                start_pos = pos

        elif state == "IN_RED":
            # abort if we clearly reverse direction upwards before reaching low_cutoff
            if de > slope_eps and e1 > e0 and e1 > high_cutoff:
                state = "IDLE"
                start_pos = None
                continue

            # successful end: we reached the lower threshold
            if e1 <= low_cutoff:
                end_pos = pos + 1
                seg_E = E[start_pos:end_pos + 1]
                seg_dE = np.diff(seg_E)

                mono_ok = False
                if len(seg_dE) > 0:
                    seg_sign = np.sign(seg_dE)
                    if slope_eps > 0:
                        seg_sign[np.abs(seg_dE) < slope_eps] = 0

                    neg_count = int(np.count_nonzero(seg_sign < 0))
                    pos_count = int(np.count_nonzero(seg_sign > 0))
                    nonzero = neg_count + pos_count

                    if nonzero > 0:
                        frac_neg = neg_count / nonzero
                        mono_ok = frac_neg >= monotonicity_fraction

                if mono_ok:
                    segments.append({
                        "kind": "reduction",
                        "start_idx": int(idx_all[start_pos]),
                        "end_idx": int(idx_all[end_pos]),
                    })

                state = "IDLE"
                start_pos = None

        elif state == "IN_OX":
            # abort if we clearly reverse direction downwards before reaching high_cutoff
            if -de > slope_eps and e1 < e0 and e1 < low_cutoff:
                state = "IDLE"
                start_pos = None
                continue

            # successful end: we reached the upper threshold
            if e1 >= high_cutoff:
                end_pos = pos + 1
                seg_E = E[start_pos:end_pos + 1]
                seg_dE = np.diff(seg_E)

                mono_ok = False
                if len(seg_dE) > 0:
                    seg_sign = np.sign(seg_dE)
                    if slope_eps > 0:
                        seg_sign[np.abs(seg_dE) < slope_eps] = 0

                    pos_count = int(np.count_nonzero(seg_sign > 0))
                    neg_count = int(np.count_nonzero(seg_sign < 0))
                    nonzero = pos_count + neg_count

                    if nonzero > 0:
                        frac_pos = pos_count / nonzero
                        mono_ok = frac_pos >= monotonicity_fraction

                if mono_ok:
                    segments.append({
                        "kind": "oxidation",
                        "start_idx": int(idx_all[start_pos]),
                        "end_idx": int(idx_all[end_pos]),
                    })

                state = "IDLE"
                start_pos = None

    return segments


# =====================================================================================
# Core: build cycles from Schmitt segments
# =====================================================================================

def split_cv_cycles(df,
                    time_col=TIME_COL_DEFAULT,
                    ctrl_col=CTRL_COL_DEFAULT,
                    cycle_col=CYCLE_COL_DEFAULT):
    """
    Use a Schmitt-trigger-like detector on control/V to define full
    reduction and oxidation sweeps.

    - Sessions: whenever time goes backwards (time/s decreases), we start a new session.
    - In each session, detect monotonic high->low (reduction) and low->high (oxidation)
      sweeps using HIGH_CUTOFF and LOW_CUTOFF.
    - Each accepted sweep (segment) becomes one global 'cycle' (row index).
    - Points not belonging to an accepted sweep remain cycle=0 and phase="unknown".

    Returns:
      df_all : original DataFrame with extra columns:
               'session', 'raw_cycle' (from EC-Lab), 'cycle', 'phase'
      df_ox  : subset with phase == "oxidation"
      df_red : subset with phase == "reduction"
    """
    print("  [2/5] Detecting cycles using Schmitt trigger on control/V...")

    # Keep original file order; don't sort by time.
    df = df.reset_index(drop=True).copy()

    # Validate required columns.
    missing = [c for c in (ctrl_col, cycle_col, time_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # numeric columns
    df[ctrl_col] = pd.to_numeric(df[ctrl_col], errors="coerce")
    df[cycle_col] = pd.to_numeric(df[cycle_col], errors="coerce")
    df["raw_cycle"] = df[cycle_col].fillna(-1).astype(int)

    # ---- detect sessions via time reset ----
    t = pd.to_numeric(df[time_col], errors="coerce")
    dt = t.diff()
    session = (dt < 0).cumsum().astype(int)
    df["session"] = session

    n_sessions = df["session"].nunique()
    print(f"      -> detected {n_sessions} session(s) via time reset")

    # Adapt Schmitt thresholds if defaults do not match the dataset's potential window.
    E_valid = df[ctrl_col].dropna()
    if E_valid.empty:
        print("      -> control/V has no valid numeric values")
        df["phase"] = "unknown"
        df["cycle"] = 0
        return df, df[df["phase"] == "oxidation"].copy(), df[df["phase"] == "reduction"].copy()

    e_low = float(E_valid.quantile(0.02))
    e_high = float(E_valid.quantile(0.98))
    span = e_high - e_low
    if span <= 0:
        e_low = float(E_valid.min())
        e_high = float(E_valid.max())
        span = e_high - e_low

    use_low = LOW_CUTOFF
    use_high = HIGH_CUTOFF
    defaults_mismatch = (HIGH_CUTOFF > float(E_valid.max())) or (LOW_CUTOFF < float(E_valid.min()))
    if defaults_mismatch and span > 0:
        use_low = e_low
        use_high = e_high
        print(f"      -> auto thresholds from data: low={use_low:.4f} V, high={use_high:.4f} V")
    else:
        print(f"      -> thresholds: low={use_low:.4f} V, high={use_high:.4f} V")

    # ---- detect Schmitt segments in each session ----
    df["phase"] = "unknown"
    df["cycle"] = 0

    all_segments = []

    for sess in sorted(df["session"].unique()):
        mask_sess = df["session"] == sess
        df_sess = df.loc[mask_sess]
        if df_sess.empty:
            continue

        segs = _detect_schmitt_segments_in_session(
            df_sess,
            ctrl_col=ctrl_col,
            high_cutoff=use_high,
            low_cutoff=use_low,
            monotonicity_fraction=MONOTONICITY_FRACTION,
        )
        all_segments.extend(segs)

    if not all_segments:
        print("      -> no valid Schmitt sweeps found")
        df_ox = df[df["phase"] == "oxidation"].copy()
        df_red = df[df["phase"] == "reduction"].copy()
        return df, df_ox, df_red

    # sort segments by starting index (across all sessions)
    all_segments_sorted = sorted(all_segments, key=lambda s: s["start_idx"])

    # assign one global cycle id per accepted sweep
    for cycle_id, seg in enumerate(all_segments_sorted, start=1):
        start_idx = seg["start_idx"]
        end_idx = seg["end_idx"]
        phase_name = "oxidation" if seg["kind"] == "oxidation" else "reduction"

        # We assume integer index is contiguous within a session
        df.loc[start_idx:end_idx, "cycle"] = cycle_id
        df.loc[start_idx:end_idx, "phase"] = phase_name

    n_cycles = len([c for c in df["cycle"].unique() if c > 0])
    n_ox = int(np.sum(df["phase"] == "oxidation"))
    n_red = int(np.sum(df["phase"] == "reduction"))

    print(f"      -> constructed {n_cycles} Schmitt-defined sweep(s)")
    print(f"         labeled points: oxidation={n_ox}, reduction={n_red}")

    df_ox = df[df["phase"] == "oxidation"].copy()
    df_red = df[df["phase"] == "reduction"].copy()
    return df, df_ox, df_red


# =====================================================================================
# Cycle stats
# =====================================================================================

def _integrate_charge(time_arr, current_mA_arr):
    """
    Integrate current (in mA) over time (in s) -> returns charge in mC.
    Trapezoidal rule.
    """
    if len(time_arr) < 2:
        return np.nan
    t = np.asarray(time_arr, dtype=float)
    I_mA = np.asarray(current_mA_arr, dtype=float)
    dt = np.diff(t)
    I_mid_mA = 0.5 * (I_mA[1:] + I_mA[:-1])

    # current in A: I_mA * 1e-3, charge in C: sum(I * dt), then mC: *1e3
    Q_C = np.sum(I_mid_mA * 1e-3 * dt)
    Q_mC = Q_C * 1e3
    return Q_mC


def compute_cycle_stats(df_all,
                        time_col=TIME_COL_DEFAULT,
                        ctrl_col=CTRL_COL_DEFAULT,
                        curr_col=Y_COL_DEFAULT,
                        filename=None):
    print("  [3/5] Computing per-cycle stats...")
    stats_rows = []
    cycles = sorted(c for c in df_all["cycle"].unique() if c > 0)

    for j, cyc in enumerate(cycles, start=1):
        df_c = df_all[df_all["cycle"] == cyc]
        if df_c.empty:
            continue

        df_ox = df_c[df_c["phase"] == "oxidation"]
        df_red = df_c[df_c["phase"] == "reduction"]

        E = df_c[ctrl_col].to_numpy(dtype=float)
        I_mA = df_c[curr_col].to_numpy(dtype=float)

        Q_ox_mC = _integrate_charge(df_ox[time_col], df_ox[curr_col]) if not df_ox.empty else np.nan
        Q_red_mC = _integrate_charge(df_red[time_col], df_red[curr_col]) if not df_red.empty else np.nan
        Q_total_mC = _integrate_charge(df_c[time_col], df_c[curr_col])

        sess = int(df_c["session"].iloc[0]) if "session" in df_c.columns else 0
        raw_cyc = int(df_c["raw_cycle"].iloc[0]) if "raw_cycle" in df_c.columns else int(cyc)

        row = dict(
            global_cycle=int(cyc),
            session=int(sess),
            raw_cycle=int(raw_cyc),
            filename=filename if filename is not None else "",
            n_points=int(len(df_c)),
            n_points_ox=int(len(df_ox)),
            n_points_red=int(len(df_red)),
            E_min=float(np.nanmin(E)),
            E_max=float(np.nanmax(E)),
            I_min_mA=float(np.nanmin(I_mA)),
            I_max_mA=float(np.nanmax(I_mA)),
            I_mean_mA=float(np.nanmean(I_mA)),
            Q_ox_mC=float(Q_ox_mC) if not np.isnan(Q_ox_mC) else np.nan,
            Q_red_mC=float(Q_red_mC) if not np.isnan(Q_red_mC) else np.nan,
            Q_total_mC=float(Q_total_mC) if not np.isnan(Q_total_mC) else np.nan,
        )
        stats_rows.append(row)

    if not stats_rows:
        print("      -> no valid cycles for stats")
        return pd.DataFrame(columns=[
            "global_cycle", "session", "raw_cycle", "filename",
            "n_points", "n_points_ox", "n_points_red",
            "E_min", "E_max",
            "I_min_mA", "I_max_mA", "I_mean_mA",
            "Q_ox_mC", "Q_red_mC", "Q_total_mC",
        ])

    stats_df = pd.DataFrame(stats_rows).set_index("global_cycle")
    print(f"      -> stats done for {len(stats_df)} cycle(s)")
    return stats_df


# =====================================================================================
# Build matrices for JSON (like df_cvloop_red / df_cvloop_ox)
# =====================================================================================

def build_cycle_matrices(df_all,
                         x_col=X_COL_DEFAULT,
                         y_col=Y_COL_DEFAULT,
                         step=EXTRACT_X_STEPSIZE):
    """
    Build interpolated matrices for oxidation and reduction.

    Returns:
      df_red:  rows = global_cycle, cols = x_grid (potentials), data = I(mA) for reduction
      df_ox:   rows = global_cycle, cols = x_grid (potentials), data = I(mA) for oxidation
      x_grid:  numpy array of x values used as columns
    """
    print("  [4/5] Building interpolated matrices for red/ox...")

    df_valid = df_all[df_all["cycle"] > 0].copy()
    if df_valid.empty:
        print("      -> no valid cycles, matrices will be empty")
        return pd.DataFrame(), pd.DataFrame(), np.array([])

    x_min = float(df_valid[x_col].min())
    x_max = float(df_valid[x_col].max())

    if x_max <= x_min:
        print(f"      -> degenerate x-range ({x_min} == {x_max}), matrices will be trivial")
        x_grid = np.array([x_min])
    else:
        step = abs(float(step))
        x_grid = np.arange(x_min, x_max + 0.5 * step, step)
        x_grid = np.round(x_grid, 6)

    cycles = sorted(df_valid["cycle"].unique())
    print(f"      -> potential range: [{x_min:.3f}, {x_max:.3f}] V, step={step}")
    print(f"      -> x_grid length: {len(x_grid)}")
    print(f"      -> cycles to interpolate: {len(cycles)}")

    df_red = pd.DataFrame(index=cycles, columns=x_grid, dtype=float)
    df_ox = pd.DataFrame(index=cycles, columns=x_grid, dtype=float)

    for j, cyc in enumerate(cycles, start=1):
        if j % 50 == 0:
            print(f"         interpolating cycle {j}/{len(cycles)}")

        df_c = df_valid[df_valid["cycle"] == cyc]

        # Oxidation
        df_ox_c = df_c[df_c["phase"] == "oxidation"].copy()
        if not df_ox_c.empty:
            xx = df_ox_c[x_col].to_numpy(dtype=float)
            yy = df_ox_c[y_col].to_numpy(dtype=float)

            sort_idx = np.argsort(xx)
            xx_sorted = xx[sort_idx]
            yy_sorted = yy[sort_idx]
            xx_unique, idx_unique = np.unique(xx_sorted, return_index=True)
            yy_unique = yy_sorted[idx_unique]

            if len(xx_unique) >= MIN_POINTS_PER_ROW:
                f_ox = interpolate.interp1d(
                    xx_unique, yy_unique,
                    bounds_error=False, fill_value=np.nan, assume_sorted=True
                )
                y_grid_ox = f_ox(x_grid)
                valid_ox = np.count_nonzero(~np.isnan(y_grid_ox))
                if valid_ox >= MIN_POINTS_PER_ROW:
                    df_ox.loc[cyc] = y_grid_ox

        # Reduction
        df_red_c = df_c[df_c["phase"] == "reduction"].copy()
        if not df_red_c.empty:
            xx = df_red_c[x_col].to_numpy(dtype=float)
            yy = df_red_c[y_col].to_numpy(dtype=float)

            sort_idx = np.argsort(xx)
            xx_sorted = xx[sort_idx]
            yy_sorted = yy[sort_idx]
            xx_unique, idx_unique = np.unique(xx_sorted, return_index=True)
            yy_unique = yy_sorted[idx_unique]

            if len(xx_unique) >= MIN_POINTS_PER_ROW:
                f_red = interpolate.interp1d(
                    xx_unique, yy_unique,
                    bounds_error=False, fill_value=np.nan, assume_sorted=True
                )
                y_grid_red = f_red(x_grid)
                valid_red = np.count_nonzero(~np.isnan(y_grid_red))
                if valid_red >= MIN_POINTS_PER_ROW:
                    df_red.loc[cyc] = y_grid_red

    # Drop rows that are completely NaN (short/empty cycles)
    before_ox = len(df_ox)
    before_red = len(df_red)
    df_ox = df_ox.dropna(how="all")
    df_red = df_red.dropna(how="all")
    after_ox = len(df_ox)
    after_red = len(df_red)

    print("      -> interpolation done")
    print(f"         oxidation cycles kept: {after_ox}, dropped (too short): {before_ox - after_ox}")
    print(f"         reduction cycles kept: {after_red}, dropped (too short): {before_red - after_red}")

    return df_red, df_ox, x_grid


# =====================================================================================
# Per-file processing
# =====================================================================================

def process_txt_file(txt_path):
    """
    Process a single EC-Lab TXT file:
      - detect cycles via Schmitt-trigger on control/V
      - compute cycle stats
      - build matrices for JSON
      - write JSON / CSV outputs into out_<safe_basename> folder
    """

    print(f"\nProcessing TXT: {txt_path}")
    df_raw = read_cv_txt(txt_path)

    df_all, df_ox_pts, df_red_pts = split_cv_cycles(
        df_raw,
        time_col=TIME_COL_DEFAULT,
        ctrl_col=CTRL_COL_DEFAULT,
        cycle_col=CYCLE_COL_DEFAULT,
    )

    basename = os.path.basename(txt_path)
    safe_name = make_safe_filename(basename)
    base_dir = os.path.dirname(txt_path) or "."
    out_dir = os.path.join(base_dir, f"out_{safe_name}")

    json_red_path = os.path.join(out_dir, "df_red.json")
    json_ox_path = os.path.join(out_dir, "df_ox.json")
    json_stats_path = os.path.join(out_dir, "cycle_stats.json")

    print("  Output directory:", out_dir)

    if (not force_reprocessing) and os.path.isfile(json_red_path) and os.path.isfile(json_ox_path):
        print("  -> JSON already exists and force_reprocessing=False, skipping.")
        return

    # 3) stats
    stats_df = compute_cycle_stats(
        df_all,
        time_col=TIME_COL_DEFAULT,
        ctrl_col=CTRL_COL_DEFAULT,
        curr_col=Y_COL_DEFAULT,
        filename=basename,
    )

    # 4) matrices
    df_red, df_ox, x_grid = build_cycle_matrices(df_all)

    # 5) save outputs
    print("  [5/5] Saving CSV and JSON outputs...")
    ensure_dir(json_red_path)

    df_red.to_csv(os.path.join(out_dir, "df_red.csv"))
    df_ox.to_csv(os.path.join(out_dir, "df_ox.csv"))
    stats_df.to_csv(os.path.join(out_dir, "cycle_stats.csv"))

    df_red.to_json(json_red_path)
    df_ox.to_json(json_ox_path)
    stats_df.to_json(json_stats_path)

    print(f"      -> wrote {json_red_path}")
    print(f"      -> wrote {json_ox_path}")
    print(f"      -> wrote {json_stats_path}")
    print("  Done with", basename)


# =====================================================================================
# Main: scan directory for TXT files
# =====================================================================================

def scan_and_process_txt(root="."):
    """
    Scan 'root' (non-recursive) for *.txt files and process them.
    """
    pattern = os.path.join(root, "*.txt")
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        print(f"No TXT files found in {root}")
        return

    print(f"Found {len(txt_files)} TXT file(s) in {root}")
    for path in txt_files:
        process_txt_file(path)


if __name__ == "__main__":
    # Change "." to a specific directory if needed
    scan_and_process_txt(".")
