import io, os, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

WINDOW_SIZE = 7
POLYNOM_ORDER = 3

def baseline_reduction(y):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    def baseline_als(y, lam, p, niter=100):
        y = np.asarray(y, dtype=float).ravel()
        L = y.size
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        DTD = lam * (D @ D.T)
        for _ in range(niter):
            W = sparse.diags(w, 0)         # (L,L)
            z = spsolve(W + DTD, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    lam = 10_000.0
    p = 0.005
    y = np.asarray(y, dtype=float).ravel()
    baseline = baseline_als(y, lam, p)
    baselined = y - baseline
    return baselined, baseline


def pre_process(wavenumbers, spectrum):
    w = np.asarray(wavenumbers, dtype=float).ravel()
    y = np.asarray(spectrum, dtype=float).ravel()

    baselined, baseline = baseline_reduction(y)

    # SavGol guards
    N = baselined.size
    win = min(WINDOW_SIZE if WINDOW_SIZE % 2 == 1 else WINDOW_SIZE + 1, N if N % 2 == 1 else N - 1)
    win = max(3, win)           # at least 3, odd
    poly = min(POLYNOM_ORDER, win - 1)

    smooth = savgol_filter(baselined, window_length=win, polyorder=poly, mode="interp")
    return w, smooth, baseline


def load_data(file_like_or_path):
    df = pd.read_csv(file_like_or_path)
    required = {"wavelength", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}, found {set(df.columns)}")
    wn = df["wavelength"].to_numpy(dtype=float)
    spectrum = df["value"].to_numpy(dtype=float)
    return wn, spectrum


st.set_page_config(page_title="Spectrum Baseline GUI", layout="wide")
st.title("Spectrum Baseline Viewer")

st.write("Upload a spectrum → shows **Raw**, **Baseline**, and **Baseline-Reduced Spectrum**.")

uploaded = st.file_uploader("Upload spectrum file", type=None, label_visibility="visible")

def call_load_data(_uploaded):
    """Supports load_data(file-like) or load_data(path)"""
    try:
        _uploaded.seek(0)
        wn, spec = load_data(_uploaded)
        return np.asarray(wn, float), np.asarray(spec, float)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(_uploaded.getbuffer())
        path = tmp.name
    try:
        wn, spec = load_data(path)
        return np.asarray(wn, float), np.asarray(spec, float)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def call_pre_process(wn, y):
    """
    Expectation: pre_process(wn, y) -> (wn_pp, y_pp, baseline)
    """
    wn_pp, y_pp, baseline = pre_process(wn, y)
    return (np.asarray(wn_pp, float),
            np.asarray(y_pp, float),
            np.asarray(baseline, float))

if uploaded:
    try:
        wn_raw, y_raw = call_load_data(uploaded)
        wn_pp, y_pp, baseline = call_pre_process(wn_raw, y_raw)

        # Get data
        # wn_raw, y_raw = load_data(uploaded)                     # raw spectrum
        # wn_pp, y_pp, baseline = pre_process(wn_raw, y_raw)      # processed + baseline

        # Plot exactly as returned
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wn_raw, y=y_raw, name="Raw", mode="lines"))
        fig.add_trace(go.Scatter(x=wn_pp, y=baseline, name="Baseline", mode="lines", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=wn_pp, y=y_pp, name="Baseline-Reduced", mode="lines"))

        fig.update_layout(height=560,
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Intensity",
                        legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)


    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info("Waiting for a file…")
