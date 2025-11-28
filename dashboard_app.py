import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Athlete Bucket Analysis")

# Force White Background, Remove Whitespace, Style Selectboxes
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #000000; }
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        footer {visibility: hidden;}
        div[data-baseweb="select"] > div { border-color: #cccccc; }
        h1, h2, h3, p, label { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA PROCESSING FUNCTIONS ---

def get_bucket(rank):
    """Maps a numeric rank to a bucket string."""
    try:
        r = int(rank)
        if r >= 31:
            return 'Rank 31 +'
        
        # Calculate bucket group (e.g., 1-3, 4-6)
        group_idx = (r - 1) // 3
        start = group_idx * 3 + 1
        end = start + 2
        return f"Rank {start} - {end}"
    except (ValueError, TypeError):
        return None

@st.cache_data
def load_and_process_data(csv_filename):
    # Check if file exists
    if not os.path.exists(csv_filename):
        return pd.DataFrame(), {}, [], 0

    df = None
    # Try multiple encodings
    for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']:
        try:
            df = pd.read_csv(csv_filename, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        st.error(f"Could not read {csv_filename}. Please check file encoding.")
        return pd.DataFrame(), {}, [], 0

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean string columns
    for col in ['season', 'competition', 'category', 'athlete']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Define Allowed Categories
    ALLOWED_CATEGORIES = [
        'Junior Men', 'Junior Women',
        'U23 Men', 'U23 Women',
        'Senior Men', 'Senior Women'
    ]

    COLORS_MEN = ['#475569', '#3b82f6', '#7dd3fc']   # Slate, Blue, Sky
    COLORS_WOMEN = ['#4a144d', '#ce73ce', '#f2d4f2'] # Dark Purple, Orchid, Light Pink

    # Initialize Datasets Structure
    # Key: (Season, Competition, Category)
    datasets = {}
    
    processed_athletes = []
    total_loaded = 0

    # Process Rows
    for idx, row in df.iterrows():
        # Get core fields
        raw_cat = row.get('category', '')
        season = row.get('season', 'Unknown')
        competition = row.get('competition', 'Unknown')
        athlete_name = row.get('athlete', f'Athlete {idx}')

        # Strict Category Matching (Case Insensitive)
        dashboard_cat = None
        for allowed in ALLOWED_CATEGORIES:
            if str(raw_cat).lower() == allowed.lower():
                dashboard_cat = allowed
                break
        
        if not dashboard_cat: continue

        # Initialize dataset key if new
        dataset_key = (season, competition, dashboard_cat)
        if dataset_key not in datasets:
            is_women = 'Women' in dashboard_cat
            datasets[dataset_key] = {
                "main": {
                    'Rank 1 - 3': [0, 0, 0], 'Rank 4 - 6': [0, 0, 0], 'Rank 7 - 9': [0, 0, 0],
                    'Rank 10 - 12': [0, 0, 0], 'Rank 13 - 15': [0, 0, 0], 'Rank 16 - 18': [0, 0, 0],
                    'Rank 19 - 21': [0, 0, 0], 'Rank 22 - 24': [0, 0, 0], 'Rank 25 - 27': [0, 0, 0],
                    'Rank 28 - 30': [0, 0, 0]
                },
                "outlier": {'Rank 31 +': [0, 0, 0]},
                "colors": COLORS_WOMEN if is_women else COLORS_MEN,
                "range": [0, 40] if is_women else [0, 80],
                "top_performers": {'Rank 1 - 3': {'Single': [], 'Double': [], 'Triple': []}, 
                                   'Rank 4 - 6': {'Single': [], 'Double': [], 'Triple': []}}
            }

        # Extract Results
        results = {}
        for cls in ['c1', 'k1', 'x1']:
            if cls in row:
                val = str(row[cls]).strip()
                if val and val != '-' and val.lower() != 'nan':
                    try:
                        results[cls.upper()] = int(float(val))
                    except ValueError:
                        pass
        
        if not results: continue

        total_loaded += 1

        # Determine Type
        count = len(results)
        type_idx = count - 1 # 0=Single, 1=Double, 2=Triple
        if type_idx < 0: type_idx = 0
        if type_idx > 2: type_idx = 2
        category_name = ['Single', 'Double', 'Triple'][type_idx]

        seen_buckets = set()
        bucket_text_map = {}

        for cls, rank in results.items():
            bucket = get_bucket(rank)
            if not bucket: continue

            if bucket not in bucket_text_map: bucket_text_map[bucket] = []
            bucket_text_map[bucket].append(f"<b>{cls}:</b> {rank}")

            if bucket not in seen_buckets:
                target = datasets[dataset_key]['outlier'] if bucket == 'Rank 31 +' else datasets[dataset_key]['main']
                if bucket in target:
                    target[bucket][type_idx] += 1
                seen_buckets.add(bucket)
                
                # Top Performers Logic
                if bucket in ['Rank 1 - 3', 'Rank 4 - 6']:
                    entry_text = f"{athlete_name}"
                    if entry_text not in datasets[dataset_key]['top_performers'][bucket][category_name]:
                        datasets[dataset_key]['top_performers'][bucket][category_name].append(entry_text)

        highlights = []
        for b, texts in bucket_text_map.items():
            highlights.append({"rank_group": b, "text": "<br>".join(texts)})

        processed_athletes.append({
            "id": idx,
            "name": athlete_name,
            "season": season,
            "competition": competition,
            "category": dashboard_cat,
            "category_index": type_idx,
            "category_name": category_name,
            "highlights": highlights
        })

    return df, datasets, processed_athletes, total_loaded

# --- 3. LOAD DATA ---
# Ensure correct filename
RAW_DF, DATASETS, ATHLETES, TOTAL_LOADED = load_and_process_data("world_championship_bucket_data.csv")

MAIN_CATEGORIES = [
    'Rank 1 - 3', 'Rank 4 - 6', 'Rank 7 - 9', 'Rank 10 - 12', 
    'Rank 13 - 15', 'Rank 16 - 18', 'Rank 19 - 21', 'Rank 22 - 24', 
    'Rank 25 - 27', 'Rank 28 - 30'
]
OUTLIER_CATEGORIES = ['Rank 31 +']

# --- 4. HELPER FUNCTIONS ---
def get_chart_arrays(data_dict, keys):
    singles = [data_dict[k][0] for k in keys]
    doubles = [data_dict[k][1] for k in keys]
    triples = [data_dict[k][2] for k in keys]
    return singles, doubles, triples

# --- 5. APP LAYOUT ---

st.title("Athlete Bucket Analysis")

if TOTAL_LOADED == 0:
    st.error("No data loaded. Please ensure 'world_championship_bucket_data.csv' is in the same directory as this script.")
    st.stop()

# --- FILTERS ROW 1: Season & Competition ---
col_season, col_comp, col_cat, col_ath = st.columns(4)

with col_season:
    # Get unique seasons sorted
    seasons = sorted(list(set(k[0] for k in DATASETS.keys())))
    selected_season = st.selectbox("Season", seasons, index=None, placeholder="Select Season")

with col_comp:
    # Filter competitions based on selected season
    if selected_season:
        competitions = sorted(list(set(k[1] for k in DATASETS.keys() if k[0] == selected_season)))
    else:
        competitions = []
    selected_competition = st.selectbox("Competition", competitions, index=None, placeholder="Select Competition")

with col_cat:
    # Filter categories based on season AND competition
    if selected_season and selected_competition:
        categories = sorted(list(set(k[2] for k in DATASETS.keys() if k[0] == selected_season and k[1] == selected_competition)))
        # Ensure logical sort order if possible
        order = ["Junior Men", "Junior Women", "U23 Men", "U23 Women", "Senior Men", "Senior Women"]
        sorted_cats = [c for c in order if c in categories] + [c for c in categories if c not in order]
    else:
        sorted_cats = []
    
    selected_category = st.selectbox("Category", sorted_cats, index=None, placeholder="Select Category")

with col_ath:
    # Filter athletes based on Season, Competition, AND Category
    if selected_season and selected_competition and selected_category:
        filtered_athletes = sorted(
            [a for a in ATHLETES if a['season'] == selected_season and a['competition'] == selected_competition and a['category'] == selected_category],
            key=lambda x: x['name']
        )
        athlete_map = {a["name"]: a for a in filtered_athletes}
        options = list(athlete_map.keys())
    else:
        athlete_map = {}
        options = []
        
    selected_name = st.selectbox("Athlete", options, index=None, placeholder="Select Athlete") 
    selected_athlete = athlete_map.get(selected_name)

# --- 6. CHECK FOR ALL SELECTIONS ---

# Only proceed if ALL selections are made
if not (selected_season and selected_competition and selected_category and selected_athlete):
    st.info("Please select a Season, Competition, Category, and Athlete to view the analysis.")
    st.stop()

# --- 7. DATA SELECTION ---

# Construct key to fetch dataset
current_key = (selected_season, selected_competition, selected_category)

if current_key not in DATASETS:
    st.info("No data available for this specific combination.")
    st.stop()

dataset = DATASETS[current_key]
active_main = dataset["main"]
active_outlier = dataset["outlier"]
active_colors = dataset["colors"]
outlier_range = dataset["range"]
top_performers = dataset["top_performers"]

st.markdown(f"<h2 style='text-align: center; text-decoration: underline; margin: 20px 0; color: black;'>{selected_category} Results {selected_season} {selected_competition}</h2>", unsafe_allow_html=True)

# --- 8. PLOTLY CHART ---

fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=False, horizontal_spacing=0.08)

c_s, c_d, c_t = active_colors

# Main Traces
s1, d1, t1 = get_chart_arrays(active_main, MAIN_CATEGORIES)

# Custom Hover Logic for Top Ranks
hover_templates_s = []
hover_templates_d = []
hover_templates_t = []

if not selected_athlete:
    for cat in MAIN_CATEGORIES:
        if cat in ['Rank 1 - 3', 'Rank 4 - 6']:
            # Single
            perf_s = top_performers.get(cat, {}).get('Single', [])
            if perf_s:
                disp_s = "<br>".join(perf_s[:15]) + ("<br>..." if len(perf_s)>15 else "")
                hover_text = f"<span style='color:{c_s}; font-weight:bold; font-size:14px'>SINGLE</span><br><br><span style='font-size:14px; font-weight:900; color:#000000'>{cat}</span><br><span style='color:#000000'>{disp_s}</span><extra></extra>"
                hover_templates_s.append(hover_text)
            else:
                hover_templates_s.append(None)
            
            # Double
            perf_d = top_performers.get(cat, {}).get('Double', [])
            if perf_d:
                disp_d = "<br>".join(perf_d[:15]) + ("<br>..." if len(perf_d)>15 else "")
                hover_text = f"<span style='color:{c_d}; font-weight:bold; font-size:14px'>DOUBLE</span><br><br><span style='font-size:14px; font-weight:900; color:#000000'>{cat}</span><br><span style='color:#000000'>{disp_d}</span><extra></extra>"
                hover_templates_d.append(hover_text)
            else:
                hover_templates_d.append(None)

            # Triple
            perf_t = top_performers.get(cat, {}).get('Triple', [])
            if perf_t:
                disp_t = "<br>".join(perf_t[:15]) + ("<br>..." if len(perf_t)>15 else "")
                hover_text = f"<span style='color:{c_t}; font-weight:bold; font-size:14px'>TRIPLE</span><br><br><span style='font-size:14px; font-weight:900; color:#000000'>{cat}</span><br><span style='color:#000000'>{disp_t}</span><extra></extra>"
                hover_templates_t.append(hover_text)
            else:
                hover_templates_t.append(None)
        else:
            hover_templates_s.append(None)
            hover_templates_d.append(None)
            hover_templates_t.append(None)
else:
    hover_templates_s = [None] * len(MAIN_CATEGORIES)
    hover_templates_d = [None] * len(MAIN_CATEGORIES)
    hover_templates_t = [None] * len(MAIN_CATEGORIES)

# Add Traces
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=s1, name='Single', marker_color=c_s, opacity=0.8, hovertemplate=hover_templates_s, hoverlabel=dict(bgcolor="white")), row=1, col=1)
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=d1, name='Double', marker_color=c_d, opacity=0.8, hovertemplate=hover_templates_d, hoverlabel=dict(bgcolor="white")), row=1, col=1)
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=t1, name='Triple', marker_color=c_t, opacity=0.8, hovertemplate=hover_templates_t, hoverlabel=dict(bgcolor="white")), row=1, col=1)

# Outlier Traces
s2, d2, t2 = get_chart_arrays(active_outlier, OUTLIER_CATEGORIES)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=s2, name='Single', marker_color=c_s, showlegend=False, opacity=0.8, hoverinfo='none'), row=1, col=2)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=d2, name='Double', marker_color=c_d, showlegend=False, opacity=0.8, hoverinfo='none'), row=1, col=2)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=t2, name='Triple', marker_color=c_t, showlegend=False, opacity=0.8, hoverinfo='none'), row=1, col=2)

# --- 9. ATHLETE SELECTION HOVER LOGIC ---

if selected_athlete:
    cat_idx = selected_athlete['category_index'] # 0=Single, 1=Double, 2=Triple
    
    # Define Colors for Text
    color_map = {0: c_s, 1: c_d, 2: c_t}
    bucket_color = color_map.get(cat_idx, "#000000")
    
    x_shift_val_main = {0: -20, 1: 0, 2: 20}.get(cat_idx, 0)
    x_shift_val_outlier = {0: -40, 1: 0, 2: 40}.get(cat_idx, 0) 

    style_config = dict(
        bgcolor="#ffffff", bordercolor="#e0e0e0", borderwidth=1,
        arrowcolor="black", opacity=1, font=dict(color="black", size=12)
    )

    for highlight in selected_athlete['highlights']:
        rank_group = highlight['rank_group']
        result_text = highlight['text']
        
        is_outlier = rank_group in OUTLIER_CATEGORIES
        target_row, target_col = (1, 2) if is_outlier else (1, 1)
        current_shift = x_shift_val_outlier if is_outlier else x_shift_val_main
        
        try:
            data_source = active_outlier if is_outlier else active_main
            bar_height = data_source[rank_group][cat_idx]

            bucket_name = selected_athlete['category_name'].upper()

            fig.add_annotation(
                x=rank_group, y=bar_height,
                text=f"<span style='font-size:12px; font-weight:bold; color:{bucket_color}'>{bucket_name}</span><br>"
                     f"<b><span style='color:black'>{selected_athlete['name']}</span></b><br>"
                     f"<span style='font-size:10px; color:#555'>{rank_group}</span><br><br>"
                     f"<span style='color:black'>{result_text}</span>",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                xshift=current_shift, ax=0, ay=-80, align="left",
                row=target_row, col=target_col, **style_config
            )
        except KeyError:
            pass

# --- 10. FINAL LAYOUT ---

fig.update_layout(
    barmode='group', bargap=0.35, bargroupgap=0, height=700,
    legend=dict(
        orientation="h", yanchor="top", y=-0.4, xanchor="center", x=0.5,
        font=dict(color="black", size=14), itemsizing='constant'
    ),
    margin=dict(b=220, t=80, l=80, r=40),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(255,255,255,1)',
    font=dict(family="Arial, sans-serif", color="black")
)

axis_config = dict(
    showline=False, gridcolor='#e5e5e5',
    tickfont=dict(color='black', size=11, weight="bold"),
    title_font=dict(color='black', size=14, weight="bold")
)

# Y-Axis (Left)
fig.update_yaxes(title_text="Athletes Per Rank Category", title_standoff=40, range=[0, 9.5], dtick=1, row=1, col=1, **axis_config)

# X-Axes
fig.update_xaxes(tickangle=-45, title_standoff=40, row=1, col=1, **axis_config)
fig.update_xaxes(tickangle=-45, title_standoff=40, row=1, col=2, **axis_config)

# Y-Axis (Right) - Fixed Range
fig.update_yaxes(range=outlier_range, dtick=10, row=1, col=2, side='left', **axis_config)

# Add Centered X-Axis Title
fig.add_annotation(
    text="Final Rank", xref="paper", yref="paper",
    x=0.5, y=-0.25, 
    showarrow=False,
    font=dict(size=16, color="black", weight="bold")
)

st.plotly_chart(fig, use_container_width=True)
