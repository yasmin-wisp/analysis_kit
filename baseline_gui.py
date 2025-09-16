import io, os, tempfile, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# -----------------------------
# Tunables
# -----------------------------
WINDOW_SIZE = 7
POLYNOM_ORDER = 3

# Color palette for unique per-spectrum colors
PALETTE = pc.qualitative.Plotly + pc.qualitative.D3 + pc.qualitative.Set3


# -----------------------------
# Baseline reduction (ALS)
# -----------------------------
def baseline_reduction(y):
    def baseline_als(y, lam, p, niter=100):
        y = np.asarray(y, dtype=float).ravel()
        L = y.size
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        DTD = lam * (D @ D.T)
        for _ in range(niter):
            W = sparse.diags(w, 0)  # (L,L)
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
    win = min(WINDOW_SIZE if WINDOW_SIZE % 2 == 1 else WINDOW_SIZE + 1,
              N if N % 2 == 1 else N - 1)
    win = max(3, win)  # at least 3, odd
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


# -----------------------------
# Helpers for session storage
# -----------------------------
def file_key(uploaded_file) -> str:
    """
    Create a stable key for a Streamlit UploadedFile using (name, size, first/last bytes).
    Avoids re-adding same file across reruns.
    """
    name = getattr(uploaded_file, "name", "unknown")
    data = uploaded_file.getbuffer()
    size = len(data)
    # hash limited bytes to avoid heavy CPU for very large files
    head = bytes(data[:4096])
    tail = bytes(data[-4096:]) if size >= 4096 else b""
    h = hashlib.md5(name.encode("utf-8") + size.to_bytes(8, "little") + head + tail).hexdigest()
    return f"{name}:{size}:{h}"


def parse_uploaded_file(uploaded_file):
    """Return (name, wn_raw, y_raw) from a Streamlit UploadedFile with robust fallback."""
    name = getattr(uploaded_file, "name", "Spectrum")
    # Try file-like read first
    try:
        uploaded_file.seek(0)
        wn, y = load_data(uploaded_file)
        return name, np.asarray(wn, float), np.asarray(y, float)
    except Exception:
        pass

    # Fallback: write to temp path then read
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spectrum Baseline GUI", layout="wide")
st.title("Spectrum Baseline Viewer")

st.write(
    "Upload one or more spectra CSVs (columns: `wavelength`, `value`). "
    "Use the buttons to toggle **Baseline-Reduced** and **Baseline** curves. "
    "You can add files incrementally — they’ll all plot on the same figure."
)

# Initialize session state
if "spectra" not in st.session_state:
    # list of dicts: {key, name, wn, y}
    st.session_state.spectra = []
if "color_map" not in st.session_state:
    # key -> color
    st.session_state.color_map = {}
if "palette_idx" not in st.session_state:
    st.session_state.palette_idx = 0
if "show_baselined_traces" not in st.session_state:
    st.session_state.show_baselined_traces = False
if "show_baseline_traces" not in st.session_state:
    st.session_state.show_baseline_traces = False

# Controls row
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
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
    if st.button("Clear all", type="secondary", use_container_width=True):
        st.session_state.spectra = []
        st.session_state.color_map = {}
        st.session_state.palette_idx = 0
        st.success("Cleared all loaded spectra.")

st.markdown("---")

# File uploader (incremental add)
new_files = st.file_uploader(
    "Upload spectrum file(s)",
    type=["csv"],
    label_visibility="visible",
    accept_multiple_files=True
)

# Append newly uploaded files into session (dedupe by key)
errors = []
if new_files:
    for uf in new_files:
        try:
            k = file_key(uf)
            if k in {s["key"] for s in st.session_state.spectra}:
                continue  # already added
            name, wn_raw, y_raw = parse_uploaded_file(uf)
            st.session_state.spectra.append({"key": k, "name": name, "wn": wn_raw, "y": y_raw})
            # Assign a color
            if k not in st.session_state.color_map:
                color = PALETTE[st.session_state.palette_idx % len(PALETTE)]
                st.session_state.color_map[k] = color
                st.session_state.palette_idx += 1
        except Exception as e:
            errors.append(f"{getattr(uf, 'name', 'Spectrum')}: {e}")

# Plot all stored spectra
if st.session_state.spectra:
    fig = go.Figure()

    for item in st.session_state.spectra:
        k = item["key"]
        name = item["name"]
        wn_raw = item["wn"]
        y_raw = item["y"]
        color = st.session_state.color_map.get(k, "#1f77b4")

        # Raw (main reference spectrum)
        fig.add_trace(
            go.Scatter(
                x=wn_raw, y=y_raw,
                name=f"{name} • Raw",
                mode="lines",
                line=dict(color=color, width=2),
                opacity=1.0
            )
        )

        # Compute processed/baseline on demand
        if st.session_state.show_baselined_traces or st.session_state.show_baseline_traces:
            wn_pp, y_pp, baseline = pre_process(wn_raw, y_raw)

            # Baseline-Reduced (lighter shade of the same color)
            if st.session_state.show_baselined_traces:
                fig.add_trace(
                    go.Scatter(
                        x=wn_pp, y=y_pp,
                        name=f"{name} • Baseline-Reduced",
                        mode="lines",
                        line=dict(color=color, width=2),
                        opacity=0.55  # lighter than Raw
                    )
                )

            # Baseline (dashed)
            if st.session_state.show_baseline_traces:
                fig.add_trace(
                    go.Scatter(
                        x=wn_pp, y=baseline,
                        name=f"{name} • Baseline",
                        mode="lines",
                        line=dict(color=color, dash="dash", width=2),
                        opacity=0.9
                    )
                )

    fig.update_layout(
        height=640,
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



############## BASIC version ####################################
# A minimal GUI to visualize baseline reduction
# using Streamlit and Plotly.
# Run with: streamlit run baseline_gui.py
#################################################################

# import io, os, tempfile
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go

# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from scipy.signal import savgol_filter

# WINDOW_SIZE = 7
# POLYNOM_ORDER = 3

# def baseline_reduction(y):
#     from scipy import sparse
#     from scipy.sparse.linalg import spsolve

#     def baseline_als(y, lam, p, niter=100):
#         y = np.asarray(y, dtype=float).ravel()
#         L = y.size
#         D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
#         w = np.ones(L)
#         DTD = lam * (D @ D.T)
#         for _ in range(niter):
#             W = sparse.diags(w, 0)         # (L,L)
#             z = spsolve(W + DTD, w * y)
#             w = p * (y > z) + (1 - p) * (y < z)
#         return z

#     lam = 10_000.0
#     p = 0.005
#     y = np.asarray(y, dtype=float).ravel()
#     baseline = baseline_als(y, lam, p)
#     baselined = y - baseline
#     return baselined, baseline


# def pre_process(wavenumbers, spectrum):
#     w = np.asarray(wavenumbers, dtype=float).ravel()
#     y = np.asarray(spectrum, dtype=float).ravel()

#     baselined, baseline = baseline_reduction(y)

#     # SavGol guards
#     N = baselined.size
#     win = min(WINDOW_SIZE if WINDOW_SIZE % 2 == 1 else WINDOW_SIZE + 1, N if N % 2 == 1 else N - 1)
#     win = max(3, win)           # at least 3, odd
#     poly = min(POLYNOM_ORDER, win - 1)

#     smooth = savgol_filter(baselined, window_length=win, polyorder=poly, mode="interp")
#     return w, smooth, baseline


# def load_data(file_like_or_path):
#     df = pd.read_csv(file_like_or_path)
#     required = {"wavelength", "value"}
#     if not required.issubset(df.columns):
#         raise ValueError(f"CSV must have columns {required}, found {set(df.columns)}")
#     wn = df["wavelength"].to_numpy(dtype=float)
#     spectrum = df["value"].to_numpy(dtype=float)
#     return wn, spectrum


# st.set_page_config(page_title="Spectrum Baseline GUI", layout="wide")
# st.title("Spectrum Baseline Viewer")

# st.write("Upload a spectrum → shows **Raw**, **Baseline**, and **Baseline-Reduced Spectrum**.")

# uploaded = st.file_uploader("Upload spectrum file", type=None, label_visibility="visible")

# def call_load_data(_uploaded):
#     """Supports load_data(file-like) or load_data(path)"""
#     try:
#         _uploaded.seek(0)
#         wn, spec = load_data(_uploaded)
#         return np.asarray(wn, float), np.asarray(spec, float)
#     except Exception:
#         pass
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(_uploaded.getbuffer())
#         path = tmp.name
#     try:
#         wn, spec = load_data(path)
#         return np.asarray(wn, float), np.asarray(spec, float)
#     finally:
#         try:
#             os.remove(path)
#         except Exception:
#             pass

# def call_pre_process(wn, y):
#     """
#     Expectation: pre_process(wn, y) -> (wn_pp, y_pp, baseline)
#     """
#     wn_pp, y_pp, baseline = pre_process(wn, y)
#     return (np.asarray(wn_pp, float),
#             np.asarray(y_pp, float),
#             np.asarray(baseline, float))

# if uploaded:
#     try:
#         wn_raw, y_raw = call_load_data(uploaded)
#         wn_pp, y_pp, baseline = call_pre_process(wn_raw, y_raw)

#         # Get data
#         # wn_raw, y_raw = load_data(uploaded)                     # raw spectrum
#         # wn_pp, y_pp, baseline = pre_process(wn_raw, y_raw)      # processed + baseline

#         # Plot exactly as returned
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=wn_raw, y=y_raw, name="Raw", mode="lines"))
#         fig.add_trace(go.Scatter(x=wn_pp, y=baseline, name="Baseline", mode="lines", line=dict(dash="dash")))
#         fig.add_trace(go.Scatter(x=wn_pp, y=y_pp, name="Baseline-Reduced", mode="lines"))

#         fig.update_layout(height=560,
#                         xaxis_title="Wavenumber (cm⁻¹)",
#                         yaxis_title="Intensity",
#                         legend_title="Series")
#         st.plotly_chart(fig, use_container_width=True)


#     except Exception as e:
#         st.error(f"Processing failed: {e}")
# else:
#     st.info("Waiting for a file…")
