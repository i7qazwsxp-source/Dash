"""
Chicago Traffic Speed Map — Dynamic Inference Dashboard
=======================================================
Run:
    pip install streamlit folium streamlit-folium pandas pytz joblib scikit-learn
    streamlit run Chicago_Dashboard_fixed.py
"""

import os
import pandas as pd
import numpy as np
import folium
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime
import pytz

from inference import TrafficPredictor

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_traffic_data.csv")
ACC_PATH  = os.path.join(BASE_DIR, "Chicago_Accidents.csv")


# Peak rush hours for demo (per Demo Strategy)
PEAK_MORNING = 8   # 7–9 AM midpoint
PEAK_EVENING = 17  # 3:30–7 PM midpoint

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chicago Traffic", page_icon="🚦", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem !important; padding-bottom: 0 !important; }

  .top-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .top-header h1 { font-family: 'DM Mono', monospace; font-size: 18px; font-weight: 500; color: #111; margin: 0; }
  .data-note { font-size: 12px; color: #999; background: #f5f5f3; border-radius: 6px; padding: 4px 10px; font-family: 'DM Mono', monospace; }

  .kpi-strip { display: grid; gap: 10px; margin-bottom: 14px; }
  .kpi { background: #fafaf8; border: 0.5px solid #e8e8e4; border-radius: 10px; padding: 12px 14px; }
  .kpi-label { font-size: 11px; color: #aaa; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 3px; }
  .kpi-val   { font-family: 'DM Mono', monospace; font-size: 24px; font-weight: 500; color: #111; line-height: 1; }
  .kpi-unit  { font-size: 12px; color: #bbb; margin-left: 3px; }
  .kpi-sub   { font-size: 11px; color: #bbb; margin-top: 4px; }

  .realtime-badge { display: inline-flex; align-items: center; gap: 6px; background: #f0faf4; border: 0.5px solid #b8e8c8; color: #2a7a4a; font-size: 12px; font-family: 'DM Mono', monospace; border-radius: 20px; padding: 4px 12px; margin-bottom: 10px; }
  .realtime-dot   { width: 7px; height: 7px; background: #2d9e4f; border-radius: 50%; animation: pulse 2s infinite; flex-shrink: 0; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .legend-row  { display: flex; align-items: center; gap: 16px; margin-top: 10px; padding: 8px 14px; background: #fafaf8; border-radius: 8px; border: 0.5px solid #e8e8e4; font-size: 12px; color: #555; }
  .legend-item { display: flex; align-items: center; gap: 7px; }
  .legend-line { width: 28px; height: 3px; border-radius: 2px; }
  .legend-sep  { width: 0.5px; height: 16px; background: #e0e0e0; }
  .legend-note { font-size: 11px; color: #bbb; margin-left: auto; font-style: italic; }

  [data-testid="stSidebar"] { background: #fafaf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Predictor (cached across reruns) ─────────────────────────────────────────
@st.cache_resource
def get_predictor():
    return TrafficPredictor(base_path=BASE_DIR)

predictor = get_predictor()

# ─── Current Chicago Time ──────────────────────────────────────────────────────
try:
    chi_tz        = pytz.timezone("America/Chicago")
    now_chi       = datetime.now(chi_tz)
except Exception:
    now_chi       = datetime.now()

current_hour    = now_chi.hour
current_day_iso = now_chi.isoweekday()       # Mon=1 … Sun=7
current_day_ds  = (current_day_iso % 7) + 1  # dataset: Sun=1 … Sat=7
is_weekday      = current_day_ds in range(2, 7)  # Mon–Fri in dataset encoding

DAYS_EN_FULL  = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}
DAYS_EN_SHORT = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}

# Demo default: peak rush hour on weekday, else morning peak
def demo_default_hour():
    if is_weekday:
        # If currently in a peak window, show current hour; else default to morning peak
        if 7 <= current_hour <= 9 or 15 <= current_hour <= 19:
            return current_hour
        return PEAK_MORNING
    return PEAK_MORNING  # weekend default still shows a busy-ish hour

default_hour = demo_default_hour()
default_day  = current_day_ds if is_weekday else 2  # fallback to Monday if weekend

# ─── Data Loader ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading traffic data…")
def load_base_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_accidents(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df_acc = pd.read_csv(path)
    df_acc.columns = df_acc.columns.str.strip().str.lower()
    if not {"latitude", "longitude", "severity"}.issubset(df_acc.columns):
        return pd.DataFrame()
    df_acc = df_acc.dropna(subset=["latitude", "longitude", "severity"])
    if "street" not in df_acc.columns:
        df_acc["street"] = "Unknown location"
    return df_acc

# ─── Traffic Slice + Fallback ──────────────────────────────────────────────────
def get_traffic_slice(df_all: pd.DataFrame, hour: int, day: int) -> pd.DataFrame:
    """
    Returns the traffic slice for the selected hour+day.
    Segments with no data for that exact hour+day get a fallback:
      1. Same day, any hour  → mean of segment on that day
      2. Same hour, any day  → mean of segment at that hour
      3. Any time            → overall segment mean
    This ensures every segment always appears on the map.
    """
    mask = (df_all["HOUR"] == hour) & (df_all["DAY_OF_WEEK"] == day)
    slice_df = df_all[mask].copy()

    # Segments present in the exact slice
    present_ids = set(slice_df["SEGMENT_ID"].unique())

    # All unique segments with their coordinate/meta info
    meta_cols = ["SEGMENT_ID", "SEGMENT_CODE", "START_LATITUDE", "START_LONGITUDE",
                 "END_LATITUDE", "END_LONGITUDE", "STREET"]
    all_meta  = df_all[meta_cols].drop_duplicates("SEGMENT_ID")
    missing   = all_meta[~all_meta["SEGMENT_ID"].isin(present_ids)]

    if missing.empty:
        return slice_df

    fallback_rows = []
    for _, row in missing.iterrows():
        sid = row["SEGMENT_ID"]
        seg_data = df_all[df_all["SEGMENT_ID"] == sid]

        # Try fallback in order
        candidate = seg_data[seg_data["DAY_OF_WEEK"] == day]
        if candidate.empty:
            candidate = seg_data[seg_data["HOUR"] == hour]
        if candidate.empty:
            candidate = seg_data

        if candidate.empty:
            continue

        # Build a representative row using mean feature values
        feature_cols = ['SEGMENT_CODE', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'WEEKEND',
                        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
                        'roll_mean_3', 'roll_mean_6', 'roll_std_3']

        fb = {col: candidate[col].mean() if col in candidate.columns else 0
              for col in feature_cols}
        fb['HOUR']        = hour
        fb['DAY_OF_WEEK'] = day
        fb['SEGMENT_ID']  = sid
        fb['START_LATITUDE']  = row['START_LATITUDE']
        fb['START_LONGITUDE'] = row['START_LONGITUDE']
        fb['END_LATITUDE']    = row['END_LATITUDE']
        fb['END_LONGITUDE']   = row['END_LONGITUDE']
        fb['STREET']          = row['STREET']
        fallback_rows.append(fb)

    if fallback_rows:
        fb_df   = pd.DataFrame(fallback_rows)
        slice_df = pd.concat([slice_df, fb_df], ignore_index=True)

    return slice_df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    st.markdown("---")
    st.markdown(
        f'<div class="realtime-badge"><span class="realtime-dot"></span>'
        f'Now — Chicago: {DAYS_EN_SHORT[current_day_ds]} {current_hour:02d}:00</div>',
        unsafe_allow_html=True,
    )

    hour = st.slider("Hour", 0, 23, value=default_hour, format="%d:00")
    day  = st.selectbox(
        "Day", options=list(DAYS_EN_FULL.keys()),
        format_func=lambda d: DAYS_EN_FULL[d],
        index=default_day - 1,
    )
    st.markdown("---")
    show_accidents = st.toggle("Show Accidents", value=False)

# ─── Load & Process ───────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(f"❌ File not found: {DATA_PATH}")
    st.stop()

df_all  = load_base_data(DATA_PATH)
traffic = get_traffic_slice(df_all, hour, day)

# Run inference
traffic["SPEED"] = predictor.predict_speed(traffic)
traffic = predictor.apply_ci_colors(traffic)

# Drop rows missing coordinates
traffic = traffic.dropna(subset=["START_LATITUDE", "START_LONGITUDE",
                                  "END_LATITUDE",   "END_LONGITUDE"])

n_segs   = len(traffic)
avg_spd  = traffic["SPEED"].mean() if n_segs else 0
heavy_pct   = (traffic["TRAFFIC_LABEL"] == "Heavy Congestion").mean() * 100 if n_segs else 0
mod_pct     = (traffic["TRAFFIC_LABEL"] == "Moderate").mean()          * 100 if n_segs else 0
free_pct    = (traffic["TRAFFIC_LABEL"] == "Free Flow").mean()          * 100 if n_segs else 0

def spd_label(s):
    return "Slow" if s < 15 else ("Moderate" if s < 30 else "Fast")

# ─── Accidents ────────────────────────────────────────────────────────────────
df_accidents = pd.DataFrame()
if show_accidents:
    df_accidents = load_accidents(ACC_PATH)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
  <h1>🚦 Chicago Traffic Map</h1>
  <span class="data-note">{DAYS_EN_FULL[day]} {hour:02d}:00 — {n_segs:,} segments</span>
</div>
""", unsafe_allow_html=True)

# ─── KPI Strip ────────────────────────────────────────────────────────────────
acc_kpi = ""
if show_accidents and not df_accidents.empty:
    sev_str = " • ".join(f"{v} {k}" for k, v in df_accidents["severity"].value_counts().items())
    acc_kpi = f"""<div class="kpi"><div class="kpi-label">⚠️ Accidents</div>
    <div class="kpi-val" style="color:#e63946">{len(df_accidents)}</div>
    <div class="kpi-sub">{sev_str}</div></div>"""

kpi_cols = "repeat(5,1fr)" if acc_kpi else "repeat(4,1fr)"

st.markdown(f"""
<div class="kpi-strip" style="grid-template-columns:{kpi_cols}">
  <div class="kpi"><div class="kpi-label">Avg Speed</div>
    <div class="kpi-val">{avg_spd:.0f}<span class="kpi-unit">mph</span></div>
    <div class="kpi-sub">{spd_label(avg_spd)}</div></div>
  <div class="kpi"><div class="kpi-label">🔴 Heavy Congestion</div>
    <div class="kpi-val" style="color:#e63946">{heavy_pct:.0f}<span class="kpi-unit">%</span></div>
    </div>
  <div class="kpi"><div class="kpi-label">🟡 Moderate</div>
    <div class="kpi-val" style="color:#f4a823">{mod_pct:.0f}<span class="kpi-unit">%</span></div>
    </div>
  <div class="kpi"><div class="kpi-label">🟢 Free Flow</div>
    <div class="kpi-val" style="color:#2d9e4f">{free_pct:.0f}<span class="kpi-unit">%</span></div>
    </div>
  {acc_kpi}
</div>
""", unsafe_allow_html=True)

if n_segs == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

# ─── Map ──────────────────────────────────────────────────────────────────────
m = folium.Map(location=[41.855, -87.66], zoom_start=11,
               tiles="CartoDB positron", control_scale=False)
m.options["attributionControl"] = False

features = [
    {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(slon), float(slat)], [float(elon), float(elat)]]
        },
        "properties": {
            "color":  clr,
            "speed":  round(float(sp), 1),
            "label":  lbl,
            "ci":     round(float(ci), 3),
            "street": str(st_name),
            "seg_id": int(sid),
        },
    }
    for slat, slon, elat, elon, st_name, sid, sp, clr, lbl, ci in zip(
        traffic["START_LATITUDE"].to_numpy(),
        traffic["START_LONGITUDE"].to_numpy(),
        traffic["END_LATITUDE"].to_numpy(),
        traffic["END_LONGITUDE"].to_numpy(),
        traffic["STREET"].to_numpy(),
        traffic["SEGMENT_ID"].to_numpy(),
        traffic["SPEED"].to_numpy(),
        traffic["TRAFFIC_COLOR"].to_numpy(),
        traffic["TRAFFIC_LABEL"].to_numpy(),
        traffic["CI"].to_numpy(),
    )
]

folium.GeoJson(
    {"type": "FeatureCollection", "features": features},
    style_function=lambda f: {
        "color": f["properties"]["color"], "weight": 2.5, "opacity": 0.82
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["street", "speed", "label"],
        aliases=["Street", "Speed (mph)", "Status"],
        style="font-size:12px;font-family:monospace;background:white;border:none;padding:4px 8px;border-radius:4px",
        sticky=True,
    ),
    popup=folium.GeoJsonPopup(
        fields=["street", "seg_id", "speed", "label"],
        aliases=["Street", "Segment", "Speed (mph)", "Status"],
        max_width=220,
    ),
    smooth_factor=1.8,
).add_to(m)

# ── Accidents Layer ───────────────────────────────────────────────────────────
if show_accidents and not df_accidents.empty:
    SEV_COLOR = {"high": "#e63946", "medium": "#f4a823", "moderate": "#f4a823", "low": "#2d9e4f"}
    sev_series = df_accidents["severity"].astype(str).str.strip().str.lower()
    acc_colors = sev_series.map(SEV_COLOR).fillna("#9b59b6")

    acc_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {"color": clr, "sev_raw": sev, "street": str(st)},
        }
        for lat, lon, sev, clr, st in zip(
            df_accidents["latitude"].to_numpy(),
            df_accidents["longitude"].to_numpy(),
            sev_series.to_numpy(),
            acc_colors.to_numpy(),
            df_accidents["street"].to_numpy(),
        )
    ]

    folium.GeoJson(
        {"type": "FeatureCollection", "features": acc_features},
        marker=folium.CircleMarker(radius=8, weight=2, fill=True, fill_opacity=0.85),
        style_function=lambda f: {
            "color": f["properties"]["color"],
            "fillColor": f["properties"]["color"],
            "fillOpacity": 0.85, "weight": 2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["sev_raw", "street"], aliases=["⚠️ Severity", "Location"],
            style="font-size:12px;font-family:monospace;background:white;border:none;padding:4px 8px;border-radius:4px",
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["street", "sev_raw"], aliases=["Street", "Severity"], max_width=220
        ),
    ).add_to(m)

st_folium(m, width="100%", height=560, returned_objects=[])

# ─── Legend ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="legend-row">
  <div class="legend-item"><div class="legend-line" style="background:#e63946"></div><span>Heavy Congestion</span></div>
  <div class="legend-sep"></div>
  <div class="legend-item"><div class="legend-line" style="background:#f4a823"></div><span>Moderate</span></div>
  <div class="legend-sep"></div>
  <div class="legend-item"><div class="legend-line" style="background:#2d9e4f"></div><span>Free Flow</span></div>
</div>
""", unsafe_allow_html=True)
