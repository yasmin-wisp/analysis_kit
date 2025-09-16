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
def load_data(spectrum_file):
    # Read the file as CSV and get the spectrum and wavelength data
    df = pd.read_csv(spectrum_file)
    if "RamanIntensity" in df.columns: # wispsense
        spectrum = df["RamanIntensity"].tolist()
    elif "value" in df.columns: # grafana
        spectrum = df["value"].tolist()
    else:
        print(f"\033[9No 'RamanIntensity' or 'value' column found in {spectrum_file}\033[0m")
        return [], []
    if "Wavelength" in df.columns: # wispsense
        wavenumbers = df["Wavelength"].tolist()
    elif "wavelength" in df.columns: # grafana
        wavenumbers = df["wavelength"].tolist()
    else:
        print(f"\033[9No 'Wavelength' or 'wavelength' column found in {spectrum_file}\033[0m")
        return [], []
    return spectrum, wavenumbers

# ==============================
# Baseline reduction (ALS)
# ==============================
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
def load_data_old(file_like_or_path):
    df = pd.read_csv(file_like_or_path)
    required = {"wavelength", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}, found {set(df.columns)}")
    wn = df["wavelength"].to_numpy(dtype=float)
    spectrum = df["value"].to_numpy(dtype=float)
    return wn, spectrum

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
    try:
        uploaded_file.seek(0)
        wn, y = load_data(uploaded_file)
        return name, np.asarray(wn, float), np.asarray(y, float)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name
    try:
        wn, y = load_data(path)
        return name, np.asarray(wn, float), np.asarray(y, float)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# ==============================
# UI / State
# ==============================
st.set_page_config(page_title="Spectrum Baseline GUI", layout="wide")
st.title("Spectrum Baseline Viewer")

# --- init session state ---
if "spectra" not in st.session_state:
    st.session_state.spectra = []          # list of {key, name, wn, y}
if "color_map" not in st.session_state:
    st.session_state.color_map = {}        # key -> color
if "palette_idx" not in st.session_state:
    st.session_state.palette_idx = 0
if "show_baselined_traces" not in st.session_state:
    st.session_state.show_baselined_traces = False
if "show_baseline_traces" not in st.session_state:
    st.session_state.show_baseline_traces = False
if "show_peaks" not in st.session_state:
    st.session_state.show_peaks = False
if "peaks_max" not in st.session_state:
    st.session_state.peaks_max = 10
if "peaks_prom_ratio" not in st.session_state:
    st.session_state.peaks_prom_ratio = 0.02
if "pending_remove" not in st.session_state:
    st.session_state.pending_remove = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0   # increment to reset the file_uploader widget

st.write(
    "Upload CSVs with columns **`wavelength`** and **`value`** or **`Wavelength`** and **`RamanIntensity`**. "
    "Toggle **Baseline-Reduced**, **Baseline**, and **Peaks (X)**. "
    "Click **✖** to remove a spectrum."
)

# --- controls ---
c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1,1])
with c1:
    if st.button("Show baselined spectrum", use_container_width=True):
        st.session_state.show_baselined_traces = True
with c2:
    if st.button("Hide baselined spectrum", use_container_width=True):
        st.session_state.show_baselined_traces = False
with c3:
    if st.button("Show the baseline", use_container_width=True):
        st.session_state.show_baseline_traces = True
with c4:
    if st.button("Hide the baseline", use_container_width=True):
        st.session_state.show_baseline_traces = False
with c5:
    if st.button("Show peaks (X)", use_container_width=True):
        st.session_state.show_peaks = True
with c6:
    if st.button("Hide peaks", use_container_width=True):
        st.session_state.show_peaks = False
with c7:
    if st.button("Clear all", use_container_width=True):
        st.session_state.spectra = []
        st.session_state.color_map = {}
        st.session_state.palette_idx = 0
        # also reset uploader so any selected files are cleared
        st.session_state.uploader_version += 1
        st.toast("Cleared all loaded spectra.")
        st.rerun()

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
    type=["csv"],
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
            name, wn_raw, y_raw = parse_uploaded_file(uf)
            st.session_state.spectra.append({"key": k, "name": name, "wn": wn_raw, "y": y_raw})
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

            col1, col2, col3 = st.columns([6, 1, 1])
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

# --- plot ---
if st.session_state.spectra:
    fig = go.Figure()
    for item in st.session_state.spectra:
        k = item["key"]; name = item["name"]
        wn_raw = item["wn"]; y_raw = item["y"]
        color = st.session_state.color_map.get(k, "#1f77b4")

        # Raw
        fig.add_trace(go.Scatter(
            x=wn_raw, y=y_raw, name=f"{name} • Raw", mode="lines",
            line=dict(color=color, width=2), opacity=1.0
        ))

        # Compute processed if needed
        need_proc = st.session_state.show_baselined_traces or st.session_state.show_baseline_traces or st.session_state.show_peaks
        wn_pp = y_pp = baseline = None
        if need_proc:
            wn_pp, y_pp, baseline = pre_process(wn_raw, y_raw)

        # Baselined (lighter)
        if st.session_state.show_baselined_traces and y_pp is not None:
            fig.add_trace(go.Scatter(
                x=wn_pp, y=y_pp, name=f"{name} • Baseline-Reduced", mode="lines",
                line=dict(color=color, width=2), opacity=0.55
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
