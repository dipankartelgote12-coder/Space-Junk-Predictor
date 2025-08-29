"""
OrbitAI – Space Junk Predictor (Streamlit, single file)

What it does
------------
• Fetch live TLEs from CelesTrak (stations / active sats / debris).
• Let the user select or paste a custom TLE.
• Propagate the orbit with SGP4 and plot an interactive 3D trajectory.
• Optional: simple ML (polynomial ridge regression) to extrapolate the path beyond physics window.
• Export the propagated / predicted ephemeris as CSV from the app.

How to run
----------
1) Save this file as `app.py`.
2) (Optional) Create a virtual env.
3) Install deps:
   pip install streamlit numpy pandas plotly requests sgp4 scikit-learn pytz
4) Start app:
   streamlit run app.py

Notes
-----
• Internet is only used to fetch TLEs from CelesTrak. You can also paste your own TLE.
• The ML layer here is a lightweight extrapolator over time on the ECI coordinates (for a demo of “AI”).
• For serious ops work, stick to physics-based propagation (SGP4) and validated catalogs.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pytz import utc

# Physics-based orbit propagation
from sgp4.api import Satrec, jday

# Lightweight "AI" extrapolator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# --------------------------
# UI CONFIG
# --------------------------
st.set_page_config(
    page_title="OrbitAI – Space Junk Predictor",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ OrbitAI – Space Junk Predictor")
st.caption(
    "Visualize and (optionally) AI-extrapolate satellite / debris orbits from Two‑Line Element (TLE) data."
)

# --------------------------
# HELPERS
# --------------------------
@dataclass
class TLE:
    name: str
    l1: str
    l2: str


CELESTRAK_SOURCES = {
    "Popular – ISS/Hubble/etc.": "https://celestrak.org/NORAD/elements/stations.txt",
    "Active Satellites": "https://celestrak.org/NORAD/elements/active.txt",
    "Starlink (subset)": "https://celestrak.org/NORAD/elements/starlink.txt",
    "Debris (bright trackable subset)": "https://celestrak.org/NORAD/elements/iridium-33-debris.txt",
}


def fetch_tles(url: str) -> List[TLE]:
    """Download and parse a CelesTrak 3-line TLE text file into TLE objects."""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    tles: List[TLE] = []
    i = 0
    while i + 2 < len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            tles.append(TLE(name=name, l1=l1, l2=l2))
            i += 3
        else:
            # Attempt to resync if the file has blank or unexpected lines
            i += 1
    return tles


def tle_to_satrec(tle: TLE) -> Satrec:
    return Satrec.twoline2rv(tle.l1, tle.l2)


def datetime_to_jday(dt: datetime) -> Tuple[int, float]:
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)


def propagate_eci(
    sat: Satrec,
    start_dt: datetime,
    minutes: int,
    step_min: float,
) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    """Propagate with SGP4; return ECI position (km), velocity (km/s), and timestamps."""
    num_steps = int(minutes / step_min) + 1
    ts = [start_dt + timedelta(minutes=step_min * k) for k in range(num_steps)]
    rs = []  # km
    vs = []  # km/s
    jd0_i, fr0 = datetime_to_jday(start_dt)
    # For each timestamp, compute (jd, fr) and propagate
    for t in ts:
        jdi, fr = datetime_to_jday(t)
        # SGP4 expects days since epoch in (jd, fr) relative to sat epoch managed internally
        e, r, v = sat.sgp4(jdi, fr)
        if e != 0:
            # Fill with NaNs if error
            r = [np.nan, np.nan, np.nan]
            v = [np.nan, np.nan, np.nan]
        rs.append(r)
        vs.append(v)
    return np.array(rs, dtype=float), np.array(vs, dtype=float), ts


def fit_ai_extrapolator(times: List[datetime], r_eci_km: np.ndarray, poly_deg: int = 3):
    """Fit three independent Polynomial Ridge models (x,y,z) vs time (seconds).
    Returns a callable f(new_times)->pred_positions.
    """
    t0 = times[0]
    tsec = np.array([(t - t0).total_seconds() for t in times]).reshape(-1, 1)
    models = []
    preds = []
    for axis in range(3):
        y = r_eci_km[:, axis]
        model = make_pipeline(PolynomialFeatures(poly_deg), Ridge(alpha=1.0))
        model.fit(tsec, y)
        models.append(model)

    def predict(new_times: List[datetime]) -> np.ndarray:
        tsec_new = np.array([(t - t0).total_seconds() for t in new_times]).reshape(-1, 1)
        out = np.zeros((len(new_times), 3), dtype=float)
        for axis in range(3):
            out[:, axis] = models[axis].predict(tsec_new)
        return out

    return predict


def make_earth_sphere(radius_km: float = 6371.0, n: int = 40):
    phi = np.linspace(0, np.pi, n)
    theta = np.linspace(0, 2 * np.pi, 2 * n)
    phi, theta = np.meshgrid(phi, theta)
    x = radius_km * np.sin(phi) * np.cos(theta)
    y = radius_km * np.sin(phi) * np.sin(theta)
    z = radius_km * np.cos(phi)
    return x, y, z


def plot_orbits(
    r_phys: Optional[np.ndarray],
    r_ai: Optional[np.ndarray],
    labels: Tuple[str, str] = ("SGP4 physics", "AI extrapolation"),
):
    fig = go.Figure()

    # Earth for context
    ex, ey, ez = make_earth_sphere()
    fig.add_surface(x=ex, y=ey, z=ez, opacity=0.15, showscale=False, name="Earth")

    if r_phys is not None and np.isfinite(r_phys).all():
        fig.add_trace(
            go.Scatter3d(
                x=r_phys[:, 0], y=r_phys[:, 1], z=r_phys[:, 2], mode="lines",
                name=labels[0],
            )
        )

    if r_ai is not None and np.isfinite(r_ai).all():
        fig.add_trace(
            go.Scatter3d(
                x=r_ai[:, 0], y=r_ai[:, 1], z=r_ai[:, 2], mode="lines",
                name=labels[1],
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
    )
    return fig


# --------------------------
# SIDEBAR – INPUTS
# --------------------------
st.sidebar.header("Data Source")
source = st.sidebar.selectbox("Choose TLE catalog", list(CELESTRAK_SOURCES.keys()))

col_a, col_b = st.sidebar.columns(2)
with col_a:
    fetch_btn = st.button("Fetch TLEs")
with col_b:
    limit = st.number_input("Max objects", min_value=1, max_value=5000, value=200, step=1)

custom_tle = st.sidebar.text_area(
    "Or paste a custom 3‑line TLE (Name + 2 lines)",
    placeholder=(
        "ISS (ZARYA)\n"
        "1 25544U 98067A   24195.51881944  .00014250  00000+0  25051-3 0  9994\n"
        "2 25544  51.6423  75.3168 0005181 280.3846 171.4356 15.49735682458338"
    ),
    height=120,
)

st.sidebar.header("Propagation")
start_time = st.sidebar.datetime_input(
    "Start (UTC)",
    value=datetime.now(timezone.utc).replace(microsecond=0),
)
phys_minutes = st.sidebar.slider("Physics window (minutes)", 10, 24 * 60, 180, step=10)
step_min = st.sidebar.slider("Step (minutes)", 1, 60, 5)

st.sidebar.header("AI Extrapolation (optional)")
ai_enable = st.sidebar.checkbox("Enable AI extrapolation beyond physics window", value=True)
ai_extra_min = st.sidebar.slider("Extra minutes via AI", 0, 24 * 60, 60, step=5)
ai_poly_deg = st.sidebar.slider("Polynomial degree", 1, 5, 3)

# --------------------------
# LOAD/SELECT TLE
# --------------------------
selected_tle: Optional[TLE] = None
fetched: List[TLE] = []

if fetch_btn:
    try:
        fetched = fetch_tles(CELESTRAK_SOURCES[source])[:limit]
        if fetched:
            st.success(f"Fetched {len(fetched)} TLEs from CelesTrak: {source}")
        else:
            st.warning("No TLEs parsed from source.")
    except Exception as e:
        st.error(f"Failed to fetch TLEs: {e}")

if fetched:
    names = [t.name for t in fetched]
    choice = st.selectbox("Select object from fetched list", names)
    selected_tle = fetched[names.index(choice)]

# Custom TLE takes priority if provided
if custom_tle.strip():
    lines = [ln for ln in custom_tle.splitlines() if ln.strip()]
    if len(lines) >= 3:
        selected_tle = TLE(name=lines[0].strip(), l1=lines[1].strip(), l2=lines[2].strip())

if not selected_tle:
    st.info("Fetch a catalog or paste a custom TLE to begin.")
    st.stop()

st.subheader(f"Selected Object: {selected_tle.name}")
with st.expander("Show TLE"):
    st.code(f"{selected_tle.name}\n{selected_tle.l1}\n{selected_tle.l2}")

# --------------------------
# PROPAGATION
# --------------------------
try:
    sat = tle_to_satrec(selected_tle)
except Exception as e:
    st.error(f"Invalid TLE: {e}")
    st.stop()

with st.spinner("Propagating orbit with SGP4…"):
    r_phys, v_phys, times_phys = propagate_eci(sat, start_time, phys_minutes, step_min)

# AI extrapolation: continue timeline beyond physics window
r_ai = None
if ai_enable and ai_extra_min > 0:
    # Fit on physics window, then predict into future
    times_ai = times_phys + [times_phys[-1] + timedelta(minutes=step_min * k)
                             for k in range(1, int(ai_extra_min / step_min) + 1)]
    try:
        extrap = fit_ai_extrapolator(times_phys, r_phys, poly_deg=ai_poly_deg)
        r_ai = extrap(times_ai)
    except Exception as e:
        st.warning(f"AI extrapolation failed: {e}")
        r_ai = None
else:
    times_ai = times_phys

# --------------------------
# VISUALIZATION
# --------------------------
fig = plot_orbits(r_phys, r_ai if ai_enable else None)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# --------------------------
# EPHEMERIS TABLE + EXPORT
# --------------------------
# Prepare dataframe for download
phys_df = pd.DataFrame({
    "timestamp_utc": times_phys,
    "x_km": r_phys[:, 0],
    "y_km": r_phys[:, 1],
    "z_km": r_phys[:, 2],
})
phys_df["source"] = "sgp4"

frames = [phys_df]
if r_ai is not None and ai_enable and len(times_ai) > len(times_phys):
    ai_df = pd.DataFrame({
        "timestamp_utc": times_ai,
        "x_km": r_ai[:, 0],
        "y_km": r_ai[:, 1],
        "z_km": r_ai[:, 2],
    })
    ai_df["source"] = "ai"
    frames.append(ai_df)

out_df = pd.concat(frames, ignore_index=True)

st.subheader("Ephemeris (ECI frame)")
st.dataframe(out_df.head(200), use_container_width=True, hide_index=True)

csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    data=csv,
    file_name=f"orbitai_{selected_tle.name.replace(' ', '_')}.csv",
    mime="text/csv",
)

st.caption(
    "ECI = Earth-Centered Inertial coordinates in kilometers. The AI path is a polynomial-time extrapolation of the physics-based points and is intended for demonstration purposes only."
)
