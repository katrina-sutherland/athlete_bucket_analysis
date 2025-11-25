import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math

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
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f9fafb;
            border-right: 1px solid #e5e7eb;
        }
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

def load_and_process_data(csv_file):
    try:
        # Try reading with utf-8, fall back if encoding issue
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin1')
            
        # Clean column names
        df.columns = df.columns.str.strip()
        
    except FileNotFoundError:
        return {}, [], {}

    # Clean category column: Title Case and Strip (e.g. " senior women " -> "Senior Women")
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).str.strip().str.title()

    # Define Allowed Categories
    ALLOWED_CATEGORIES = [
        'Junior Men', 'Junior Women',
        'U23 Men', 'U23 Women',
        'Senior Men', 'Senior Women'
    ]

    COLORS_MEN = ['#475569', '#3b82f6', '#7dd3fc']   # Slate, Blue, Sky
    COLORS_WOMEN = ['#4a144d', '#ce73ce', '#f2d4f2'] # Dark Purple, Orchid, Light Pink

    # Initialize Datasets
    datasets = {}
    counts_summary = {cat: 0 for cat in ALLOWED_CATEGORIES} # For Debugging

    for cat in ALLOWED_CATEGORIES:
        is_women = 'Women' in cat
        datasets[cat] = {
            "main": {
                'Rank 1 - 3': [0, 0, 0], 'Rank 4 - 6': [0, 0, 0], 'Rank 7 - 9': [0, 0, 0],
                'Rank 10 - 12': [0, 0, 0], 'Rank 13 - 15': [0, 0, 0], 'Rank 16 - 18': [0, 0, 0],
                'Rank 19 - 21': [0, 0, 0], 'Rank 22 - 24': [0, 0, 0], 'Rank 25 - 27': [0, 0, 0],
                'Rank 28 - 30': [0, 0, 0]
            },
            "outlier": {'Rank 31 +': [0, 0, 0]},
            "colors": COLORS_WOMEN if is_women else COLORS_MEN,
            "range": [0, 10]
        }

    processed_athletes = []

    # Process Rows
    for idx, row in df.iterrows():
        raw_cat = row.get('category', '')
        
        # Strict matching against allowed list
        if raw_cat not in ALLOWED_CATEGORIES:
            continue

        dashboard_cat = raw_cat

        # Extract Results
        results = {}
        for cls in ['C1', 'K1', 'X1']:
            if cls in row:
                val = str(row[cls]).strip()
                if val and val != '-' and val.lower() != 'nan' and val != '':
                    try:
                        results[cls] = int(float(val))
                    except ValueError:
                        pass
        
        if not results: continue

        # Update Counts
        counts_summary[dashboard_cat] += 1

        # Determine Type
        count = len(results)
        type_idx = count - 1 
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
                target = datasets[dashboard_cat]['outlier'] if bucket == 'Rank 31 +' else datasets[dashboard_cat]['main']
                if bucket in target:
                    target[bucket][type_idx] += 1
                seen_buckets.add(bucket)

        highlights = []
        for b, texts in bucket_text_map.items():
            highlights.append({"rank_group": b, "text": "<br>".join(texts)})

        processed_athletes.append({
            "id": idx,
            "name": row.get('athlete', f'Athlete {idx}'),
            "category": dashboard_cat,
            "category_index": type_idx,
            "category_name": category_name,
            "highlights": highlights
        })

    # Calculate Y-Axis Ranges
    for cat, data in datasets.items():
        if 'Women' in cat:
            data['range'] = [0, 40]
        else:
            data['range'] = [0, 80]

    return datasets, processed_athletes, counts_summary

# --- 3. LOAD DATA ---
DATASETS, ATHLETES, COUNTS = load_and_process_data("2025_wch_buckets.csv")

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

# -- SIDEBAR DATA INSPECTOR --
with st.sidebar:
    st.header("Data Inspector")
    st.markdown("Check below to ensure all data is loaded correctly from the CSV.")
    
    total_loaded = sum(COUNTS.values())
    st.metric("Total Athletes", total_loaded)
    
    st.subheader("Count per Category")
    df_counts = pd.DataFrame.from_dict(COUNTS, orient='index', columns=['Count'])
    st.dataframe(df_counts, use_container_width=True)
    
    if total_loaded == 0:
        st.error("No data loaded. Is '2025_wch_buckets.csv' in the folder?")

st.title("Athlete Bucket Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    selected_event = st.selectbox("Event", ["2025 World Championships", "2024 World Championships", "2023 World Cup"])
with col2:
    # Filter only categories that have data, or show all standard ones
    available_cats = sorted(list(DATASETS.keys()))
    selected_category = st.selectbox("Category", available_cats)
with col3:
    filtered_athletes = sorted([a for a in ATHLETES if a['category'] == selected_category], key=lambda x: x['name'])
    athlete_map = {a["name"]: a for a in filtered_athletes}
    options = ["None"] + list(athlete_map.keys())
    selected_name = st.selectbox("Athlete", options)
    selected_athlete = athlete_map.get(selected_name)

# --- 6. DATA SELECTION ---

if not DATASETS or selected_category not in DATASETS:
    st.info("Waiting for data...")
    st.stop()

dataset = DATASETS.get(selected_category)
active_main = dataset["main"]
active_outlier = dataset["outlier"]
active_colors = dataset["colors"]
outlier_range = dataset["range"]

st.markdown(f"<h2 style='text-align: center; text-decoration: underline; margin: 20px 0; color: black;'>{selected_category} Results {selected_event}</h2>", unsafe_allow_html=True)

# --- 7. PLOTLY CHART ---

fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=False, horizontal_spacing=0.08)

c_s, c_d, c_t = active_colors

# Main Traces
s1, d1, t1 = get_chart_arrays(active_main, MAIN_CATEGORIES)
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=s1, name='Single', marker_color=c_s, opacity=0.8), row=1, col=1)
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=d1, name='Double', marker_color=c_d, opacity=0.8), row=1, col=1)
fig.add_trace(go.Bar(x=MAIN_CATEGORIES, y=t1, name='Triple', marker_color=c_t, opacity=0.8), row=1, col=1)

# Outlier Traces
s2, d2, t2 = get_chart_arrays(active_outlier, OUTLIER_CATEGORIES)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=s2, name='Single', marker_color=c_s, showlegend=False, opacity=0.8), row=1, col=2)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=d2, name='Double', marker_color=c_d, showlegend=False, opacity=0.8), row=1, col=2)
fig.add_trace(go.Bar(x=OUTLIER_CATEGORIES, y=t2, name='Triple', marker_color=c_t, showlegend=False, opacity=0.8), row=1, col=2)

# --- 8. HOVER LOGIC ---

if selected_athlete:
    cat_idx = selected_athlete['category_index']
    color_map = {0: c_s, 1: c_d, 2: c_t}
    text_color = color_map.get(cat_idx, "#000000")
    
    # Default shifts for Main Chart (Rank 1-30)
    # Left for Single, Center for Double, Right for Triple
    x_shift_val_main = {0: -20, 1: 0, 2: 20}.get(cat_idx, 0)
    
    # Shifts for Outlier Chart (Rank 31+)
    # Often needs wider spacing as the single "Rank 31+" bar group can be wider or centered differently
    x_shift_val_outlier = {0: -30, 1: 0, 2: 30}.get(cat_idx, 0) 

    style_config = dict(
        bgcolor="#ffffff", bordercolor="#e0e0e0", borderwidth=1,
        arrowcolor="black", opacity=1, font=dict(color="black", size=12)
    )

    for highlight in selected_athlete['highlights']:
        rank_group = highlight['rank_group']
        result_text = highlight['text']
        
        is_outlier = rank_group in OUTLIER_CATEGORIES
        target_row, target_col = (1, 2) if is_outlier else (1, 1)
        
        # Use specific shift value based on chart type
        current_x_shift = x_shift_val_outlier if is_outlier else x_shift_val_main
        
        try:
            data_source = active_outlier if is_outlier else active_main
            bar_height = data_source[rank_group][cat_idx]

            fig.add_annotation(
                x=rank_group, y=bar_height,
                text=f"<b>{selected_athlete['name']}</b><br>"
                     f"<span style='font-size:10px; color:#555'>{rank_group} • </span>"
                     f"<span style='font-size:10px; font-weight:bold; color:{text_color}'>{selected_athlete['category_name'].upper()}</span><br><br>"
                     f"{result_text}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                xshift=current_x_shift, ax=0, ay=-80, align="left",
                row=target_row, col=target_col, **style_config
            )
        except KeyError:
            pass

# --- 9. FINAL LAYOUT ---

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

# X-Axes (No individual titles)
fig.update_xaxes(tickangle=-45, title_standoff=40, row=1, col=1, **axis_config)
fig.update_xaxes(tickangle=-45, title_standoff=40, row=1, col=2, **axis_config)

# Y-Axis (Right) - Dynamic Range
fig.update_yaxes(range=outlier_range, dtick=10, row=1, col=2, side='left', **axis_config)

# Add Centered X-Axis Title
fig.add_annotation(
    text="Final Rank", xref="paper", yref="paper",
    x=0.5, y=-0.25, 
    showarrow=False,
    font=dict(size=16, color="black", weight="bold")
)

st.plotly_chart(fig, use_container_width=True)
