import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Athlete Bucket Analysis")

# Force White Background, Remove Whitespace
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #000000; }
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        footer {visibility: hidden;}
        div[data-baseweb="select"] > div { border-color: #cccccc; }
        h1, h2, h3, p, label { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA SETUP ---

main_categories = [
    'Rank 1 - 3', 'Rank 4 - 6', 'Rank 7 - 9', 'Rank 10 - 12', 
    'Rank 13 - 15', 'Rank 16 - 18', 'Rank 19 - 21', 'Rank 22 - 24', 
    'Rank 25 - 27', 'Rank 28 - 30'
]

outlier_categories = ['Rank 31 +']

# --- DATASETS ---

# 1. Junior Men Data (Blue Theme)
jnr_men_main = {
    'Rank 1 - 3':   [2, 2, 7], 
    'Rank 4 - 6':   [5, 3, 1],
    'Rank 7 - 9':   [4, 4, 1], 
    'Rank 10 - 12': [7, 2, 0],
    'Rank 13 - 15': [3, 6, 0], 
    'Rank 16 - 18': [3, 4, 2],
    'Rank 19 - 21': [4, 3, 2], 
    'Rank 22 - 24': [2, 4, 3],
    'Rank 25 - 27': [1, 4, 4], 
    'Rank 28 - 30': [4, 2, 3]
}
jnr_men_outlier = {'Rank 31 +': [27, 27, 66]}
jnr_men_colors = ['#475569', '#3b82f6', '#7dd3fc'] # Slate, Blue, Sky
jnr_men_axis_range = [0, 80] 

# 2. Junior Women Data (Purple/Pink Theme)
jnr_women_main = {
    'Rank 1 - 3':   [1, 6, 2], 
    'Rank 4 - 6':   [4, 3, 2],
    'Rank 7 - 9':   [1, 2, 6], 
    'Rank 10 - 12': [2, 1, 6],
    'Rank 13 - 15': [1, 3, 5], 
    'Rank 16 - 18': [3, 2, 4], 
    'Rank 19 - 21': [1, 4, 4], 
    'Rank 22 - 24': [3, 3, 3], 
    'Rank 25 - 27': [2, 6, 1], 
    'Rank 28 - 30': [2, 2, 5]
}
jnr_women_outlier = {'Rank 31 +': [9, 38, 22]}
jnr_women_colors = ['#4a144d', '#ce73ce', '#f2d4f2'] # Dark Purple, Orchid, Light Pink
jnr_women_axis_range = [0, 45]

# --- ATHLETE DATABASE (UPDATED STRUCTURE) ---
# 'highlights' list allows results to appear in multiple places
athletes = [
    {
        "id": 1,
        "name": "HOCEVAR Ziga-Lin",
        "category": "Junior Men",
        "category_index": 2, # Triple
        "category_name": "Triple",
        "highlights": [
            {"rank_group": "Rank 1 - 3", "text": "<b>K1:</b> 3rd<br><b>X1:</b> 1st"},
            {"rank_group": "Rank 4 - 6", "text": "<b>C1:</b> 4th"}
        ]
    },
    {
        "id": 2,
        "name": "Example Athlete (Double)",
        "category": "Junior Men",
        "category_index": 1, 
        "category_name": "Double",
        "highlights": [
            {"rank_group": "Rank 10 - 12", "text": "<b>C1:</b> 11th<br><b>K1:</b> 12th"}
        ]
    },
    {
        "id": 4,
        "name": "DANEK Hanna",
        "category": "Junior Women",
        "category_index": 2, # Triple
        "category_name": "Triple",
        "highlights": [
            {"rank_group": "Rank 1 - 3", "text": "<b>K1:</b> 1st"},
            {"rank_group": "Rank 16 - 18", "text": "<b>X1:</b> 17th"},
            {"rank_group": "Rank 22 - 24", "text": "<b>C1:</b> 23rd"}
        ]
    },
    {
        "id": 5,
        "name": "BLUME Mina",
        "category": "Junior Women",
        "category_index": 1, # Double
        "category_name": "Double",
        "highlights": [
            {"rank_group": "Rank 1 - 3", "text": "<b>X1:</b> 1st<br><b>K1:</b> 3rd"}
        ]
    }
]

# --- 3. HELPER FUNCTIONS ---
def get_chart_arrays(data_dict, keys):
    singles = [data_dict[k][0] for k in keys]
    doubles = [data_dict[k][1] for k in keys]
    triples = [data_dict[k][2] for k in keys]
    return singles, doubles, triples

# --- 4. APP LAYOUT ---

st.title("Athlete Bucket Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    selected_event = st.selectbox("Event", ["2025 World Championships", "2024 World Championships", "2023 World Cup"])
with col2:
    selected_category = st.selectbox("Category", ["Junior Men", "Junior Women", "U23 Men", "Senior Men"])
with col3:
    athlete_map = {a["name"]: a for a in athletes}
    options = ["None"] + list(athlete_map.keys())
    selected_name = st.selectbox("Athlete", options)
    selected_athlete = athlete_map.get(selected_name)

# --- 5. DATA SELECTION LOGIC ---

if selected_category == "Junior Women":
    active_main_data = jnr_women_main
    active_outlier_data = jnr_women_outlier
    active_colors = jnr_women_colors
    outlier_range = jnr_women_axis_range
else:
    active_main_data = jnr_men_main
    active_outlier_data = jnr_men_outlier
    active_colors = jnr_men_colors
    outlier_range = jnr_men_axis_range

st.markdown(f"<h2 style='text-align: center; text-decoration: underline; margin: 20px 0; color: black;'>{selected_category} Results {selected_event}</h2>", unsafe_allow_html=True)

# --- 6. PLOTLY CHART ---

fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=False, horizontal_spacing=0.08)

# Colors extracted from selection
c_single, c_double, c_triple = active_colors

# Main Chart
s1, d1, t1 = get_chart_arrays(active_main_data, main_categories)
fig.add_trace(go.Bar(x=main_categories, y=s1, name='Single', marker_color=c_single, opacity=0.8), row=1, col=1)
fig.add_trace(go.Bar(x=main_categories, y=d1, name='Double', marker_color=c_double, opacity=0.8), row=1, col=1)
fig.add_trace(go.Bar(x=main_categories, y=t1, name='Triple', marker_color=c_triple, opacity=0.8), row=1, col=1)

# Outlier Chart
s2, d2, t2 = get_chart_arrays(active_outlier_data, outlier_categories)
fig.add_trace(go.Bar(x=outlier_categories, y=s2, name='Single', marker_color=c_single, showlegend=False, opacity=0.8), row=1, col=2)
fig.add_trace(go.Bar(x=outlier_categories, y=d2, name='Double', marker_color=c_double, showlegend=False, opacity=0.8), row=1, col=2)
fig.add_trace(go.Bar(x=outlier_categories, y=t2, name='Triple', marker_color=c_triple, showlegend=False, opacity=0.8), row=1, col=2)

# --- 7. HOVER TEMPLATE LOGIC (MULTI-LOCATION) ---

if selected_athlete:
    cat_idx = selected_athlete['category_index'] # 0, 1, or 2
    
    # Determine text color based on category
    color_map = {0: c_single, 1: c_double, 2: c_triple}
    text_color = color_map.get(cat_idx, "#000000")
    
    # Alignment Offset (Left for Single, Center for Double, Right for Triple)
    x_shift_map = {0: -20, 1: 0, 2: 20} 
    x_shift_val = x_shift_map.get(cat_idx, 0)

    # Style Config for the card
    style_config = dict(
        bgcolor="#ffffff",
        bordercolor="#e0e0e0",
        borderwidth=1,
        arrowcolor="black",
        opacity=1,
        font=dict(color="black", size=12)
    )

    # Loop through each highlight result
    for highlight in selected_athlete['highlights']:
        rank_group = highlight['rank_group']
        result_text = highlight['text']
        
        is_outlier = rank_group in outlier_categories
        target_row, target_col = (1, 2) if is_outlier else (1, 1)
        
        # Retrieve bar height to position annotation
        try:
            data_source = active_outlier_data if is_outlier else active_main_data
            bar_height = data_source[rank_group][cat_idx]

            fig.add_annotation(
                x=rank_group,
                y=bar_height,
                text=f"<b>{selected_athlete['name']}</b><br>"
                     f"<span style='font-size:10px; color:#555'>{rank_group} • </span>"
                     f"<span style='font-size:10px; font-weight:bold; color:{text_color}'>{selected_athlete['category_name'].upper()}</span><br><br>"
                     f"{result_text}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                xshift=x_shift_val,  
                ax=0, ay=-80,       
                align="left",
                row=target_row, col=target_col,
                **style_config
            )
        except KeyError:
            # If mapping is wrong or data missing, just skip that annotation
            pass

# --- 8. FINAL LAYOUT ---

fig.update_layout(
    barmode='group', bargap=0.35, bargroupgap=0, height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="black")),
    margin=dict(b=150, t=80, l=80, r=40),
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(255,255,255,1)',
    font=dict(family="Arial, sans-serif", color="black")
)

axis_config = dict(
    showline=False, gridcolor='#e5e5e5',
    tickfont=dict(color='black', size=11, weight="bold"),
    title_font=dict(color='black', size=14, weight="bold")
)

fig.update_yaxes(title_text="Athletes Per Rank Category", title_standoff=20, range=[0, 9.5], dtick=1, row=1, col=1, **axis_config)
fig.update_xaxes(title_text="Final Rank", tickangle=-45, title_standoff=20, row=1, col=1, **axis_config)

# Update Outlier Y-Axis based on selected category (range variable)
fig.update_yaxes(range=outlier_range, dtick=10, row=1, col=2, side='left', **axis_config)
fig.update_xaxes(title_text="Final Rank", tickangle=-45, title_standoff=20, row=1, col=2, **axis_config)

st.plotly_chart(fig, use_container_width=True)
