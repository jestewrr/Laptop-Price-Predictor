import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Laptop Price Prediction System", layout="wide")

# --------------- Custom CSS ---------------
st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .header-container h1 {
        color: white !important;
        margin: 0;
        font-family: 'Helvetica', sans-serif;
        letter-spacing: 1px;
    }
    .header-container p {
        color: #a0aec0;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        border: none !important;
        height: 55px;
        margin-top: 15px;
        border-radius: 8px !important;
        transition: transform 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 10px;
    }
    .metric-card h3 {
        color: #a0aec0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 8px 0;
    }
    .metric-card h2 {
        color: #e2e8f0;
        font-size: 28px;
        margin: 0;
        font-weight: 700;
    }
    .metric-card .delta-pos { color: #48bb78; font-size: 13px; }
    .metric-card .delta-neg { color: #fc8181; font-size: 13px; }
    .metric-card .delta-neutral { color: #a0aec0; font-size: 13px; }
    .section-header {
        color: #e2e8f0;
        border-left: 4px solid #0f3460;
        padding-left: 12px;
        margin: 30px 0 15px 0;
        font-size: 20px;
    }
    .specs-badge {
        display: inline-block;
        background: #0f3460;
        color: #e2e8f0;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin: 3px 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------- Helper: Google Sheets connection ---------------
def get_gsheet_client():
    """Return an authorized gspread client, or None."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None
    except Exception:
        return None
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)


DB_COLUMNS = [
    "timestamp",
    "ram_size",
    "storage_rom",
    "processor",
    "display_quality",
    "human_model_price",
    "ai_model_price",
]


def get_prediction_sheet(client):
    """Return the prediction worksheet and ensure the header row exists."""
    sheet = client.open("Laptop Price Predictions").sheet1
    first_row = sheet.row_values(1)
    if first_row != DB_COLUMNS:
        if not first_row:
            sheet.append_row(DB_COLUMNS)
        else:
            sheet.insert_row(DB_COLUMNS, 1)
    return sheet


def normalize_history_df(df):
    """Force database history into the expected schema."""
    if df.empty:
        return df

    df = df.copy()
    if list(df.columns) == DB_COLUMNS:
        return df

    if len(df.columns) >= len(DB_COLUMNS):
        df = df.iloc[:, :len(DB_COLUMNS)]
        df.columns = DB_COLUMNS
    return df

def load_history():
    """Read all rows from the Google Sheet and return a DataFrame."""
    client = get_gsheet_client()
    if client is None:
        return pd.DataFrame()
    try:
        sheet = get_prediction_sheet(client)
        rows = sheet.get_all_values()
        if len(rows) <= 1:
            return pd.DataFrame()
        data_rows = rows[1:] if rows[0] == DB_COLUMNS else rows
        df = pd.DataFrame(data_rows, columns=DB_COLUMNS)
        return normalize_history_df(df)
    except Exception:
        return pd.DataFrame()

def save_prediction(ram, rom, processor, display, price_h, price_a):
    """Append a prediction row to Google Sheets."""
    client = get_gsheet_client()
    if client is None:
        return
    try:
        sheet = get_prediction_sheet(client)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, ram, rom, processor, display,
                          round(float(price_h), 2), round(float(price_a), 2)])
    except Exception as e:
        st.warning(f"Google Sheets connection error: {e}")

# --------------- Load ML models ---------------
@st.cache_resource
def load_model():
    model_path = 'dual_price_models.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

# --------------- Header ---------------
st.markdown("""
    <div class="header-container">
        <h1>Laptop Price Prediction System</h1>
        <p>Dual-Model Machine Learning Engine &bull; Compare Human-Tuned vs AI-Optimized Predictions</p>
    </div>
""", unsafe_allow_html=True)

data = load_model()

if data is None:
    st.error("Model file 'dual_price_models.pkl' not found.")
    st.stop()

h_model = data['human_model']
a_model = data['ai_model']
le = data['label_encoder']
features = data['feature_names']

ram_options = ['8GB', '16GB', '32GB']
rom_options = ['256GB', '512GB', '1TB', '2TB']
display_options = ['FHD', '4K']
processor_options = (list(le.classes_) if hasattr(le, 'classes_')
                     else ['i3', 'i5', 'i7', 'i9',
                            'Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9'])

# --------------- Prediction function ---------------
def get_predictions(ram, rom, processor, display_q):
    input_df = pd.DataFrame(0, index=[0], columns=features)
    if f'ram_{ram}' in input_df.columns:
        input_df[f'ram_{ram}'] = 1
    if f'rom_{rom}' in input_df.columns:
        input_df[f'rom_{rom}'] = 1
    if f'display_resolution_{display_q}' in input_df.columns:
        input_df[f'display_resolution_{display_q}'] = 1
    try:
        input_df['processor_encoded'] = le.transform([processor])[0]
    except Exception:
        input_df['processor_encoded'] = 0
    price_h = h_model.predict(input_df)[0]
    price_a = a_model.predict(input_df)[0]
    return price_h, price_a

# =============== SIDEBAR – Input Controls ===============
with st.sidebar:
    st.markdown("### Select Specifications")
    ram = st.selectbox("RAM Size", ram_options)
    rom = st.selectbox("Storage (ROM)", rom_options)
    processor = st.selectbox("Processor", processor_options)
    display_q = st.selectbox("Display Quality", display_options)
    predict_clicked = st.button("Get Prediction",
                                type="primary", use_container_width=True)

# =============== MAIN CONTENT ===============
if predict_clicked or 'predictions_made' not in st.session_state:
    st.session_state['predictions_made'] = True
    price_h, price_a = get_predictions(ram, rom, processor, display_q)

    if predict_clicked:
        save_prediction(ram, rom, processor, display_q, price_h, price_a)

    # --- Spec badges ---
    st.markdown(
        f'<span class="specs-badge">RAM: {ram}</span>'
        f'<span class="specs-badge">Storage: {rom}</span>'
        f'<span class="specs-badge">CPU: {processor}</span>'
        f'<span class="specs-badge">Display: {display_q}</span>',
        unsafe_allow_html=True)

    # --- Metric cards row ---
    diff = price_h - price_a
    pct_diff = (diff / price_a * 100) if price_a != 0 else 0
    avg_price = (price_h + price_a) / 2

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Human Model</h3>
            <h2>{price_h:,.2f}</h2>
            <span class="delta-neutral">Estimated Price</span>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>AI Model</h3>
            <h2>{price_a:,.2f}</h2>
            <span class="delta-neutral">Estimated Price</span>
        </div>""", unsafe_allow_html=True)

    with m3:
        delta_class = "delta-pos" if diff >= 0 else "delta-neg"
        arrow = "▲" if diff >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Price Difference</h3>
            <h2>{abs(diff):,.2f}</h2>
            <span class="{delta_class}">{arrow} {abs(pct_diff):.2f}%</span>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Estimate</h3>
            <h2>{avg_price:,.2f}</h2>
            <span class="delta-neutral">Mean of both models</span>
        </div>""", unsafe_allow_html=True)

    # --- Charts Row ---
    st.markdown('<h3 class="section-header">Model Comparison Charts</h3>',
                unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        # Bar chart comparison
        plot_data = pd.DataFrame({
            'Model': ['Human Model', 'AI Model'],
            'Predicted Price': [price_h, price_a]
        })
        fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=100)
        sns.set_theme(style="whitegrid")
        barplot = sns.barplot(x='Model', y='Predicted Price', hue='Model',
                              data=plot_data, legend=False, ax=ax1,
                              palette=['#1A5F7A', '#DD5353'])
        ax1.set_title(f'Price Prediction Comparison\n({ram}, {rom}, {processor}, {display_q})',
                      fontsize=14, fontweight='bold', pad=12)
        ax1.set_ylabel('Estimated Price', fontsize=12)
        ax1.set_xlabel('')
        ax1.tick_params(labelsize=11)
        for p in barplot.patches:
            h = p.get_height()
            ax1.annotate(f'{h:,.2f}',
                         (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom',
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=11, fontweight='bold', color='#333')
        fig1.tight_layout()
        st.pyplot(fig1)

    with chart2:
        # Radar / Pie breakdown showing contribution split
        fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=100)
        sizes = [price_h, price_a]
        labels = [f'Human\n{price_h:,.2f}', f'AI\n{price_a:,.2f}']
        colors = ['#1A5F7A', '#DD5353']
        explode = (0.04, 0.04)
        wedges, texts, autotexts = ax2.pie(
            sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=140,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
        for t in autotexts:
            t.set_color('white')
            t.set_fontsize(12)
        ax2.set_title('Model Contribution Split', fontsize=14,
                      fontweight='bold', pad=15)
        fig2.tight_layout()
        st.pyplot(fig2)

    # --- Feature Input Summary Table ---
    st.markdown('<h3 class="section-header">Feature Input Summary</h3>',
                unsafe_allow_html=True)

    feature_df = pd.DataFrame({
        'Feature': ['RAM Size', 'Storage (ROM)', 'Processor', 'Display Quality'],
        'Selected Value': [ram, rom, processor, display_q],
        'Encoding': [
            f'ram_{ram}' if f'ram_{ram}' in features else 'Baseline (all 0s)',
            f'rom_{rom}' if f'rom_{rom}' in features else 'Baseline (all 0s)',
            f'Encoded → {le.transform([processor])[0]}' if processor in (le.classes_ if hasattr(le, "classes_") else []) else 'N/A',
            f'display_resolution_{display_q}' if f'display_resolution_{display_q}' in features else 'Baseline (all 0s)'
        ]
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# =============== PREDICTION HISTORY ===============
st.markdown('<h3 class="section-header">Prediction History (from Database)</h3>',
            unsafe_allow_html=True)

history_df = load_history()

if history_df.empty:
    st.info("No prediction history found yet. Click **Get Prediction** to start logging data.")
else:
    display_df = normalize_history_df(history_df)

    # Show summary metrics of history
    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        st.metric("Total Predictions", len(display_df))
    with hm2:
        if 'human_model_price' in display_df.columns:
            avg_h = pd.to_numeric(display_df['human_model_price'], errors='coerce').mean()
            st.metric("Avg Human Price", f"{avg_h:,.2f}" if not pd.isna(avg_h) else "N/A")
    with hm3:
        if 'ai_model_price' in display_df.columns:
            avg_a = pd.to_numeric(display_df['ai_model_price'], errors='coerce').mean()
            st.metric("Avg AI Price", f"{avg_a:,.2f}" if not pd.isna(avg_a) else "N/A")

    # History chart: line plot of predictions over time
    if 'timestamp' in display_df.columns and len(display_df) >= 2:
        st.markdown('<h3 class="section-header">Price Trend Over Time</h3>',
                    unsafe_allow_html=True)
        trend_df = display_df.copy()
        trend_df['human_model_price'] = pd.to_numeric(trend_df['human_model_price'], errors='coerce')
        trend_df['ai_model_price'] = pd.to_numeric(trend_df['ai_model_price'], errors='coerce')
        trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'], errors='coerce')

        fig3, ax3 = plt.subplots(figsize=(10, 5), dpi=100, facecolor='white')
        ax3.set_facecolor('white')

        # Human Model – orange line with outlined circle markers
        ax3.plot(trend_df['timestamp'], trend_df['human_model_price'],
                 color='#FF8C00', linewidth=3, zorder=2)
        ax3.scatter(trend_df['timestamp'], trend_df['human_model_price'],
                    s=100, facecolors='#FFA500', edgecolors='#CC7000',
                    linewidths=2, zorder=3, label='Human Model')

        # AI Model – magenta/pink line with outlined circle markers
        ax3.plot(trend_df['timestamp'], trend_df['ai_model_price'],
                 color='#FF1493', linewidth=3, zorder=2)
        ax3.scatter(trend_df['timestamp'], trend_df['ai_model_price'],
                    s=100, facecolors='#FF69B4', edgecolors='#C71585',
                    linewidths=2, zorder=3, label='AI Model')

        ax3.set_xlabel('Date & Time', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Price', fontsize=13, fontweight='bold')
        fig3.autofmt_xdate(rotation=45)
        ax3.set_title('Prediction History Trend', fontsize=16, fontweight='bold', pad=12)
        ax3.legend(fontsize=12, frameon=True, fancybox=True, shadow=True,
                   loc='upper left')
        ax3.grid(True, alpha=0.35, color='#cccccc', linestyle='-')
        ax3.tick_params(axis='both', labelsize=11)
        for spine in ax3.spines.values():
            spine.set_visible(False)
        fig3.tight_layout()
        st.pyplot(fig3)

    # Full history table
    st.dataframe(display_df, use_container_width=True, hide_index=True)