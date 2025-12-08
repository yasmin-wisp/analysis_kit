import io, os, tempfile, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, find_peaks, peak_prominences

# ==============================
# Tunables
# ==============================
WINDOW_SIZE = 7
POLYNOM_ORDER = 3
PALETTE = pc.qualitative.Plotly + pc.qualitative.D3 + pc.qualitative.Set3  # stable per-file colors

# ==============================
from typing import Tuple, Union, Optional
from pathlib import Path

def read_lightnovo_tsv(
    tsv_path: Union[str, Path],
    metadata_rows: int = 8,
    return_df: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read ONE LightNovo TSV file and return:
      spectrum_df : DataFrame with ['Wavelength','RamanIntensity'] OR 
      > wn      : numpy array of wavenumbers
      > spectra : numpy array of Raman intensities
      metadata_df : DataFrame containing the first metadata_rows lines as-is
    """

    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"File not found: {tsv_path}")

    # Read the entire raw TSV
    df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str, engine="python")
    if df.empty:
        raise ValueError(f"Empty TSV file: {tsv_path}")

    # --- Metadata (top rows) ---
    metadata_raw = df.iloc[:metadata_rows, :2].copy()   # take only first two cols
    metadata_raw = metadata_raw.reset_index(drop=True)

    # Extract keys (column names) and values (row 0)
    keys = metadata_raw.iloc[:, 0].astype(str).str.strip().tolist()
    vals = metadata_raw.iloc[:, 1].astype(str).str.strip().tolist()

    # Build one-row DataFrame where:
    #   columns = metadata keys
    #   row 0   = metadata values
    meta_df = pd.DataFrame([vals], columns=keys)

    # --- Spectrum (remaining rows) ---
    if df.shape[0] <= metadata_rows or df.shape[1] < 2:
        raise ValueError("No spectral data in this file")

    spectra_data = df.iloc[metadata_rows:].copy()

    if return_df:
        spectrum_df = pd.DataFrame({
            "Wavelength": pd.to_numeric(spectra_data.iloc[:, 0], errors="coerce"),
            "RamanIntensity": pd.to_numeric(spectra_data.iloc[:, 1], errors="coerce")
        }).dropna()

        spectrum_df["Wavelength"] = spectrum_df["Wavelength"].astype(int)

        return spectrum_df, meta_df
    wn = pd.to_numeric(spectra_data.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    spectra = pd.to_numeric(spectra_data.iloc[:, 1], errors="coerce").dropna().to_numpy(dtype=float)

    return wn, spectra, meta_df

from pathlib import Path

def load_data(spectrum_file):
    path = Path(spectrum_file)
    ext = path.suffix.lower()

    if ext == ".tsv":
        df = read_lightnovo_tsv(path, return_df=True)[0]
    else:
        df = pd.read_csv(path)  # standard CSV

    if "RamanIntensity" in df.columns:       # wispsense
        spectrum = df["RamanIntensity"].tolist()
    elif "value" in df.columns:              # grafana
        spectrum = df["value"].tolist()
    elif "avg_main_normalized" in df.columns:
        spectrum = df["avg_main_normalized"].tolist()
    elif "avg_main" in df.columns:
        spectrum = df["avg_main"].tolist()
    else:
        print(f"\033[9No 'RamanIntensity' or 'value' or 'avg_main' column found in {spectrum_file}\033[0m")
        return [], []

    if "Wavelength" in df.columns:
        wavenumbers = df["Wavelength"].tolist()
    elif "wavelength" in df.columns:
        wavenumbers = df["wavelength"].tolist()
    elif "raman_shift_cm-1" in df.columns:
        wavenumbers = df["raman_shift_cm-1"].tolist()
    else:
        print(f"\033[9No 'Wavelength' or 'wavelength' or 'raman_shift_cm-1' column found in {spectrum_file}\033[0m")
        return [], []

    # Check for reference spectrum
    reference = None
    if "avg_ref_raw" in df.columns:
        reference = df["avg_ref_raw"].tolist()
    elif "avg_ref" in df.columns:
        reference = df["avg_ref"].tolist()

    # Check for raw spectrum
    raw = None
    if "avg_main_raw" in df.columns:
        raw = df["avg_main_raw"].tolist()

    return wavenumbers, spectrum, reference, raw


# def load_data(spectrum_file):
#     # Read the file as CSV and get the spectrum and wavelength data
#     if spectrum_file.endswith(".tsv"):
#         df, meta = read_lightnovo_tsv(spectrum_file, return_df=True)
#         return read_lightnovo_tsv(spectrum_file)[:2]
#     else:
#         df = pd.read_csv(spectrum_file)
#     if "RamanIntensity" in df.columns: # wispsense
#         spectrum = df["RamanIntensity"].tolist()
#     elif "value" in df.columns: # grafana
#         spectrum = df["value"].tolist()
#     else:
#         print(f"\033[9No 'RamanIntensity' or 'value' column found in {spectrum_file}\033[0m")
#         return [], []
#     if "Wavelength" in df.columns: # wispsense
#         wavenumbers = df["Wavelength"].tolist()
#     elif "wavelength" in df.columns: # grafana
#         wavenumbers = df["wavelength"].tolist()
#     else:
#         print(f"\033[9No 'Wavelength' or 'wavelength' column found in {spectrum_file}\033[0m")
#         return [], []
#     return wavenumbers, spectrum

# ==============================
# Baseline reduction (ALS)
# ==============================
def build_baselined_csv(wn_raw, y_raw) -> str:
    # Compute processed spectrum
    wn_pp, y_pp, _ = pre_process(wn_raw, y_raw)
    # Save in the same schema you read: wavelength/value (lowercase)
    df = pd.DataFrame({"wavelength": wn_pp, "value": y_pp})
    return df.to_csv(index=False)

def baseline_reduction(y):
    def baseline_als(y, lam, p, niter=100):
        y = np.asarray(y, dtype=float).ravel()
        L = y.size
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        DTD = lam * (D @ D.T)
        for _ in range(niter):
            W = sparse.diags(w, 0)
            z = spsolve(W + DTD, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    lam = 10_000.0
    p = 0.005
    y = np.asarray(y, float).ravel()
    baseline = baseline_als(y, lam, p)
    baselined = y - baseline
    return baselined, baseline

def pre_process(wavenumbers, spectrum):
    w = np.asarray(wavenumbers, float).ravel()
    y = np.asarray(spectrum, float).ravel()
    baselined, baseline = baseline_reduction(y)

    N = baselined.size
    win = min(WINDOW_SIZE if WINDOW_SIZE % 2 == 1 else WINDOW_SIZE + 1,
              N if N % 2 == 1 else N - 1)
    win = max(3, win)
    poly = min(POLYNOM_ORDER, win - 1)

    smooth = savgol_filter(baselined, window_length=win, polyorder=poly, mode="interp")
    return w, smooth, baseline

# ==============================
# Peaks
# ==============================
def detect_peaks(x, y, max_peaks=10, prom_ratio=0.02):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return np.array([]), np.array([])

    y = np.nan_to_num(y, copy=False)
    span = float(np.max(y) - np.min(y))
    if not np.isfinite(span) or span <= 0:
        return np.array([]), np.array([])

    min_prom = max(prom_ratio * span, 1e-12)
    idxs, props = find_peaks(y, prominence=min_prom)
    if idxs.size == 0:
        return np.array([]), np.array([])

    prominences = props.get("prominences")
    if prominences is None:
        prominences, _, _ = peak_prominences(y, idxs)

    order = np.argsort(prominences)[::-1]
    idxs = idxs[order[:max_peaks]]
    return x[idxs], y[idxs]

# ==============================
# I/O helpers
# ==============================
# def load_data_old(file_like_or_path):
#     df = pd.read_csv(file_like_or_path)
#     required = {"wavelength", "value"}
#     if not required.issubset(df.columns):
#         raise ValueError(f"CSV must have columns {required}, found {set(df.columns)}")
#     wn = df["wavelength"].to_numpy(dtype=float)
#     spectrum = df["value"].to_numpy(dtype=float)
#     return wn, spectrum

def file_key(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "unknown")
    data = uploaded_file.getbuffer()
    size = len(data)
    head = bytes(data[:4096])
    tail = bytes(data[-4096:]) if size >= 4096 else b""
    h = hashlib.md5(name.encode("utf-8") + size.to_bytes(8, "little") + head + tail).hexdigest()
    return f"{name}:{size}:{h}"

def parse_uploaded_file(uploaded_file):
    name = getattr(uploaded_file, "name", "Spectrum")
    _, ext = os.path.splitext(name)  # ".csv" or ".tsv"

    # Always write to temp with the same suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name

    try:
        wn, y, ref, raw = load_data(path)
        return name, np.asarray(wn, float), np.asarray(y, float), np.asarray(ref, float) if ref is not None else None, np.asarray(raw, float) if raw is not None else None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# def parse_uploaded_file(uploaded_file):
#     name = getattr(uploaded_file, "name", "Spectrum")
#     try:
#         uploaded_file.seek(0)
#         wn, y = load_data(uploaded_file)
#         return name, np.asarray(wn, float), np.asarray(y, float)
#     except Exception:
#         pass
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(uploaded_file.getbuffer())
#         path = tmp.name
#     try:
#         wn, y = load_data(path)
#         return name, np.asarray(wn, float), np.asarray(y, float)
#     finally:
#         try:
#             os.remove(path)
#         except Exception:
#             pass

# ==============================
# UI / State
# ==============================

import io, base64
from PIL import Image

LOGO_PATH = "Small_Wisp_Logo_Blue_OuterGlow.png"   # <- put your logo file here

@st.cache_data
def load_logo_bytes(path: str = LOGO_PATH) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# Load once for both favicon and page header
_logo_bytes = load_logo_bytes()
_logo_img = Image.open(io.BytesIO(_logo_bytes))

# Use the logo as the browser tab icon
st.set_page_config(page_title="Spectrum Baseline GUI", page_icon=_logo_img, layout="wide")

# --- header with logo
left, right = st.columns([1, 0.2], vertical_alignment="center")
with left:
    st.title("Spectrum Baseline Viewer")
with right:
    st.image(_logo_img, width=140)


# --- init session state ---
if "spectra" not in st.session_state:
    st.session_state.spectra = []          # list of {key, name, wn, y}
if "color_map" not in st.session_state:
    st.session_state.color_map = {}        # key -> color
if "palette_idx" not in st.session_state:
    st.session_state.palette_idx = 0
if "show_raw_traces" not in st.session_state:
    st.session_state.show_raw_traces = True
if "show_baselined_traces" not in st.session_state:
    st.session_state.show_baselined_traces = False
if "show_baseline_traces" not in st.session_state:
    st.session_state.show_baseline_traces = False
if "perform_savgol" not in st.session_state:
    st.session_state.perform_savgol = False
if "perform_minmax_normalization" not in st.session_state:
    st.session_state.perform_minmax_normalization = False
if "show_peaks" not in st.session_state:
    st.session_state.show_peaks = False
if "peaks_max" not in st.session_state:
    st.session_state.peaks_max = 10
if "peaks_prom_ratio" not in st.session_state:
    st.session_state.peaks_prom_ratio = 0.02
if "show_reference" not in st.session_state:
    st.session_state.show_reference = False
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False
if "pending_remove" not in st.session_state:
    st.session_state.pending_remove = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0   # increment to reset the file_uploader widget

st.write(
    "Upload TSV or CSVs with columns **`wavelength`** or **`Wavelength`** or **`raman_shift_cm-1`** AND **`value`** or **`RamanIntensity`** or **`avg_main`**. \n"
    "Toggle **Baseline-Reduced**, **Baseline**, and **Peaks (X)**. "
    "Click **✖** to remove a spectrum."
)

# --- controls ---
with st.expander("Display options", expanded=True):
    st.checkbox("Show original spectrum", key="show_raw_traces") #, value=st.session_state.get("show_raw_traces", True))
    st.checkbox("Show baselined spectrum", key="show_baselined_traces", value=st.session_state.get("show_baselined_traces", False))
    st.checkbox("Show baseline", key="show_baseline_traces", value=st.session_state.get("show_baseline_traces", False))
    st.checkbox("Show peaks (X)", key="show_peaks", value=st.session_state.get("show_peaks", False))
    st.checkbox("Apply Savitzky-Golay smoothing (after baseline)", key="perform_savgol", value=st.session_state.get("perform_savgol", False))
    st.checkbox("Apply Min-Max normalization (after baseline)", key="perform_minmax_normalization", value=st.session_state.get("perform_minmax_normalization", False))
    show_ref_checkbox = st.checkbox("Show reference spectrum", key="show_reference", value=st.session_state.get("show_reference", False))
    show_raw_checkbox = st.checkbox("Show raw spectrum", key="show_raw", value=st.session_state.get("show_raw", False))

    # Note about reference availability
    if show_ref_checkbox and st.session_state.spectra:
        has_reference = any(item.get("ref") is not None for item in st.session_state.spectra)
        if not has_reference:
            st.caption("⚠️ Reference does not exist in the loaded files")

    # Note about raw availability
    if show_raw_checkbox and st.session_state.spectra:
        has_raw = any(item.get("raw") is not None for item in st.session_state.spectra)
        if not has_raw:
            st.caption("⚠️ Raw does not exist in the loaded files")


with st.expander("Peaks settings"):
    st.session_state.peaks_max = st.slider("Max peaks per spectrum", 1, 50, st.session_state.peaks_max, 1)
    st.session_state.peaks_prom_ratio = st.slider(
        "Prominence ratio (relative to signal range)", 0.001, 0.2,
        float(st.session_state.peaks_prom_ratio), 0.001
    )

st.markdown("---")

# --- APPLY PENDING REMOVAL EARLY (and reset uploader) ---
rk = st.session_state.get("pending_remove")
if rk:
    st.session_state.spectra = [s for s in st.session_state.spectra if s["key"] != rk]
    st.session_state.color_map.pop(rk, None)
    st.session_state.pending_remove = None
    # Force the uploader to clear its selection so the removed file isn't re-added
    st.session_state.uploader_version += 1
    st.rerun()

# --- uploader (incremental add) ---
new_files = st.file_uploader(
    "Upload spectrum file(s)",
    type=["csv", "tsv"],
    label_visibility="visible",
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_version}"   # 👈 changes on delete/clear
)

# Append new files (dedupe by key)
errors = []
if new_files:
    existing_keys = {s["key"] for s in st.session_state.spectra}
    for uf in new_files:
        try:
            k = file_key(uf)
            if k in existing_keys:
                continue
            name, wn_raw, y_raw, ref_raw, raw_raw = parse_uploaded_file(uf)
            st.session_state.spectra.append({"key": k, "name": name, "wn": wn_raw, "y": y_raw, "ref": ref_raw, "raw": raw_raw})
            if k not in st.session_state.color_map:
                color = PALETTE[st.session_state.palette_idx % len(PALETTE)]
                st.session_state.color_map[k] = color
                st.session_state.palette_idx += 1
        except Exception as e:
            errors.append(f"{getattr(uf, 'name', 'Spectrum')}: {e}")

# --- list & remove controls ---
if st.session_state.spectra:
    with st.expander("Loaded spectra", expanded=True):
        for item in list(st.session_state.spectra):
            k = item["key"]; name = item["name"]
            npts = int(item["wn"].size)
            color = st.session_state.color_map.get(k, "#1f77b4")

            # col1, col2, col3 = st.columns([6, 1, 1])
            col1, col2, col3, col4 = st.columns([5, 1, 1, 2])

            with col1:
                st.markdown(f"**{name}** · {npts} pts")
            with col2:
                st.markdown(
                    f"<div style='width:18px;height:18px;border-radius:4px;background:{color};border:1px solid #999;'></div>",
                    unsafe_allow_html=True
                )
            with col3:
                if st.button("✖", key=f"rm_{k}", help="Remove this spectrum", use_container_width=True):
                    st.session_state.pending_remove = k
                    st.rerun()  # quick visual feedback
            with col4:
                base, _ = os.path.splitext(name)
                out_name = f"{base}_baselined.csv"
                try:
                    csv_str = build_baselined_csv(item["wn"], item["y"])
                    st.download_button(
                        "Save baselined CSV",
                        data=csv_str,
                        file_name=out_name,
                        mime="text/csv",
                        key=f"dl_{k}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.caption(f"Baselined export unavailable: {e}")

# --- plot ---
if st.session_state.spectra:
    fig = go.Figure()
    for item in st.session_state.spectra:
        k = item["key"]; name = item["name"]
        wn_raw = item["wn"]; y_raw = item["y"]; ref_raw = item.get("ref"); raw_raw = item.get("raw")
        color = st.session_state.color_map.get(k, "#1f77b4")

        # Raw
        if st.session_state.show_raw_traces:
            fig.add_trace(go.Scatter(
                x=wn_raw, y=y_raw, name=name, mode="lines",
                line=dict(color=color, width=2), opacity=1.0
            ))

        # Reference spectrum
        if st.session_state.show_reference and ref_raw is not None:
            fig.add_trace(go.Scatter(
                x=wn_raw, y=ref_raw, name=f"{name} • Reference", mode="lines",
                line=dict(color=color, dash="dot", width=2), opacity=0.8
            ))

        # Raw spectrum
        if st.session_state.show_raw and raw_raw is not None:
            fig.add_trace(go.Scatter(
                x=wn_raw, y=raw_raw, name=f"{name} • Raw", mode="lines",
                line=dict(color=color, dash="dashdot", width=2), opacity=0.7
            ))

        # Compute processed if needed
        need_proc = st.session_state.show_baselined_traces or st.session_state.show_baseline_traces or st.session_state.show_peaks
        wn_pp = y_pp = baseline = None
        if need_proc:
            wn_pp, y_pp, baseline = pre_process(wn_raw, y_raw)

        # Baselined (lighter)
        if st.session_state.show_baselined_traces and y_pp is not None:
            # Normalization
            if st.session_state.perform_minmax_normalization:
                y_min = np.min(y_pp)
                y_max = np.max(y_pp)
                if y_max > y_min:
                    y_pp = (y_pp - y_min) / (y_max - y_min)
            if st.session_state.perform_savgol:
                smooth_baselined_spectrum = savgol_filter(y_pp, 7, 3)
                fig.add_trace(go.Scatter(
                    x=wn_pp, y=smooth_baselined_spectrum, name=f"{name} • Baseline-Reduced (Smoothed)", mode="lines",
                    line=dict(color=color, width=2), opacity=0.75
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=wn_pp, y=y_pp, name=f"{name} • Baseline-Reduced", mode="lines",
                    line=dict(color=color, width=2), opacity=0.55
                ))

        # Compute savgol smoothing if needed
        if st.session_state.perform_savgol and not st.session_state.show_baselined_traces:
            smooth_spectrum = savgol_filter(y_raw, 7, 3)
            fig.add_trace(go.Scatter(
                x=wn_raw, y=smooth_spectrum, name=f"{name} • Smoothed", mode="lines",
                line=dict(color=color, width=2), opacity=0.75
            ))


        # Baseline (dashed)
        if st.session_state.show_baseline_traces and baseline is not None:
            fig.add_trace(go.Scatter(
                x=wn_pp, y=baseline, name=f"{name} • Baseline", mode="lines",
                line=dict(color=color, dash="dash", width=2), opacity=0.9
            ))

        # Peaks (X)
        if st.session_state.show_peaks and y_pp is not None:
            px, py = detect_peaks(wn_pp, y_pp,
                                  max_peaks=st.session_state.peaks_max,
                                  prom_ratio=st.session_state.peaks_prom_ratio)
            if px.size > 0:
                fig.add_trace(go.Scatter(
                    x=px, y=py, name=f"{name} • Peaks", mode="markers",
                    marker=dict(symbol="x", size=10, line=dict(width=2), color=color),
                    opacity=0.95,
                    hovertemplate="Wavenumber: %{x:.2f} cm⁻¹<br>Intensity: %{y:.2f}<extra></extra>"
                ))

    fig.update_layout(
        height=700,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity",
        legend_title="Series",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    if errors:
        st.error("One or more files failed to process:\n\n- " + "\n- ".join(errors))
else:
    st.info("Waiting for file(s)…")
