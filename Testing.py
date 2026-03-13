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

def save_prediction(ram, rom, processor, display, price_h):
    """Append a prediction row to Google Sheets."""
    client = get_gsheet_client()
    if client is None:
        return
    try:
        sheet = get_prediction_sheet(client)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, ram, rom, processor, display,
                          round(float(price_h), 2), round(float(price_h), 2)])
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

data = load_model()

if data is None:
    st.error("Model file 'dual_price_models.pkl' not found.")
    st.stop()

try:
    h_model = data.get('human_model')
    le = data.get('label_encoder') or data.get('processor_encoder')
    features = list(getattr(h_model, 'feature_names_in_', [])) or list(data.get('feature_names', []))
    
    if h_model is None:
        st.error("Model data corrupted: human_model (Random Forest) not found in pickle file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

ram_options = ['8GB', '16GB', '32GB']
rom_options = ['256GB', '512GB', '1TB', '2TB']
display_options = ['FHD', '4K']

def normalize_feature_token(value):
    return ''.join(ch.lower() for ch in str(value) if ch.isalnum())


def convert_spec_to_gb(value):
    value = str(value).upper()
    if 'TB' in value:
        return int(value.replace('TB', '')) * 1024
    if 'GB' in value:
        return int(value.replace('GB', ''))
    return 0


def derive_processor_options(feature_columns, processor_encoder):
    if processor_encoder is not None and hasattr(processor_encoder, 'classes_'):
        return list(processor_encoder.classes_)

    derived = []
    for col in feature_columns:
        col_str = str(col)
        if col_str.startswith('processor_') and col_str != 'processor_encoded':
            label = col_str.split('processor_', 1)[1].replace('_', ' ')
            if label and label not in derived:
                derived.append(label)

    return derived or ['i3', 'i5', 'i7', 'i9', 'Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9']


processor_options = derive_processor_options(features, le)


def build_input_frame(ram, rom, processor, display_q):
    input_df = pd.DataFrame(0.0, index=[0], columns=features)
    feature_lookup = {normalize_feature_token(col): col for col in features}
    encoding_notes = {
        'RAM Size': [],
        'Storage (ROM)': [],
        'Processor': [],
        'Display Quality': [],
    }

    def assign_feature(normalized_name, value, note_bucket, note_label):
        actual_name = feature_lookup.get(normalized_name)
        if actual_name is not None:
            input_df.at[0, actual_name] = value
            encoding_notes[note_bucket].append(note_label)
            return True
        return False

    ram_token = normalize_feature_token(ram)
    rom_token = normalize_feature_token(rom)
    processor_token = normalize_feature_token(processor)
    display_token = normalize_feature_token(display_q)

    ram_gb = convert_spec_to_gb(ram)
    rom_gb = convert_spec_to_gb(rom)

    assign_feature('ramgb', ram_gb, 'RAM Size', 'ram_gb')
    assign_feature('ramnumeric', ram_gb, 'RAM Size', 'ram_numeric')
    assign_feature('ram', ram_gb, 'RAM Size', 'ram')
    assign_feature(f'ram{ram_token}', 1.0, 'RAM Size', f'ram_{ram}')

    assign_feature('romgb', rom_gb, 'Storage (ROM)', 'rom_gb')
    assign_feature('storagegb', rom_gb, 'Storage (ROM)', 'storage_gb')
    assign_feature('storage', rom_gb, 'Storage (ROM)', 'storage')
    assign_feature('rom', rom_gb, 'Storage (ROM)', 'rom')
    assign_feature(f'rom{rom_token}', 1.0, 'Storage (ROM)', f'rom_{rom}')
    assign_feature(f'storage{rom_token}', 1.0, 'Storage (ROM)', f'storage_{rom}')

    assign_feature('displayresolution', 1.0 if display_q == '4K' else 0.0, 'Display Quality', 'display_resolution')
    assign_feature('displayquality', 1.0 if display_q == '4K' else 0.0, 'Display Quality', 'display_quality')
    assign_feature('display', 1.0 if display_q == '4K' else 0.0, 'Display Quality', 'display')
    assign_feature(f'displayresolution{display_token}', 1.0, 'Display Quality', f'display_resolution_{display_q}')
    assign_feature(f'display{display_token}', 1.0, 'Display Quality', f'display_{display_q}')

    processor_dummy_set = assign_feature(f'processor{processor_token}', 1.0, 'Processor', f'processor_{processor}')

    if le is not None and hasattr(le, 'transform'):
        try:
            encoded_val = float(le.transform([processor])[0])
            if assign_feature('processorencoded', encoded_val, 'Processor', 'processor_encoded'):
                processor_dummy_set = True
            assign_feature('processor', encoded_val, 'Processor', 'processor')
        except Exception:
            pass

    for key, notes in encoding_notes.items():
        if not notes:
            encoding_notes[key] = 'Baseline / no matching feature column'
        else:
            encoding_notes[key] = ', '.join(notes)

    if not processor_dummy_set and encoding_notes['Processor'] == 'Baseline / no matching feature column':
        encoding_notes['Processor'] = 'No processor feature match found in exported model'

    return input_df, encoding_notes

# --------------- Prediction function ---------------
def get_predictions(ram, rom, processor, display_q):
    try:
        if not features:
            st.error("Feature names not available in model.")
            return None
        input_df, _ = build_input_frame(ram, rom, processor, display_q)
        price = h_model.predict(input_df)[0]
        return price
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

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
    predicted_price = get_predictions(ram, rom, processor, display_q)

    if predicted_price is None:
        st.stop()
    
    if predict_clicked:
        save_prediction(ram, rom, processor, display_q, predicted_price)

    # --- Spec badges ---
    st.markdown(
        f'<span class="specs-badge">RAM: {ram}</span>'
        f'<span class="specs-badge">Storage: {rom}</span>'
        f'<span class="specs-badge">CPU: {processor}</span>'
        f'<span class="specs-badge">Display: {display_q}</span>',
        unsafe_allow_html=True)

    # --- Single metric card for prediction ---
    st.markdown(f"""
    <div class="metric-card" style="border: 2px solid #48bb78;">
        <h3>Predicted Price (Random Forest)</h3>
        <h2>₹{predicted_price:,.2f}</h2>
        <span class="delta-pos">Best Estimate</span>
    </div>""", unsafe_allow_html=True)

    # --- Feature Input Summary Table ---
    st.markdown('<h3 class="section-header">Feature Input Summary</h3>',
                unsafe_allow_html=True)

    _, encoding_notes = build_input_frame(ram, rom, processor, display_q)

    feature_df = pd.DataFrame({
        'Feature': ['RAM Size', 'Storage (ROM)', 'Processor', 'Display Quality'],
        'Selected Value': [ram, rom, processor, display_q],
        'Encoding': [
            encoding_notes['RAM Size'],
            encoding_notes['Storage (ROM)'],
            encoding_notes['Processor'],
            encoding_notes['Display Quality']
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
    hm1, hm2 = st.columns(2)
    with hm1:
        st.metric("Total Predictions", len(display_df))
    with hm2:
        if 'human_model_price' in display_df.columns:
            avg_p = pd.to_numeric(display_df['human_model_price'], errors='coerce').mean()
            st.metric("Avg Predicted Price", f"{avg_p:,.2f}" if not pd.isna(avg_p) else "N/A")

    # History chart: line plot of predictions over time
    if 'timestamp' in display_df.columns and len(display_df) >= 2:
        st.markdown('<h3 class="section-header">Price Prediction Trend</h3>',
                    unsafe_allow_html=True)
        trend_df = display_df.copy()
        trend_df['human_model_price'] = pd.to_numeric(trend_df['human_model_price'], errors='coerce')
        trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'], errors='coerce')

        fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=100, facecolor='#f5f7fa')
        ax3.set_facecolor('#ffffff')

        # Plot Random Forest Model only
        ax3.plot(trend_df['timestamp'], trend_df['human_model_price'],
                 color='#3498db', linewidth=3.5, zorder=3, label='Random Forest Model',
                 marker='o', markersize=8, markerfacecolor='#5dade2', 
                 markeredgecolor='#2c3e50', markeredgewidth=1.5, alpha=0.95)
        
        # Add subtle area fill under the line
        ax3.fill_between(trend_df['timestamp'], 
                         trend_df['human_model_price'],
                         alpha=0.15, color='#3498db', zorder=1)

        # Enhanced styling
        ax3.set_xlabel('Date & Time', fontsize=13, fontweight='bold', color='#2c3e50', labelpad=10)
        ax3.set_ylabel('Predicted Price (₹)', fontsize=13, fontweight='bold', color='#2c3e50', labelpad=10)
        fig3.autofmt_xdate(rotation=45, ha='right')
        
        ax3.set_title('Random Forest: Laptop Price Predictions Over Time', 
                      fontsize=16, fontweight='bold', pad=20, color='#1a1a2e', 
                      fontfamily='sans-serif', loc='left')
        
        # Premium legend styling
        ax3.legend(fontsize=11, frameon=True, fancybox=True, shadow=True,
                   loc='upper left', framealpha=0.95, edgecolor='#bdc3c7', 
                   borderpad=1.2, labelspacing=0.8)
        
        # Refined grid
        ax3.grid(True, alpha=0.2, color='#bdc3c7', linestyle='--', linewidth=0.8)
        ax3.set_axisbelow(True)
        
        # Enhanced tick styling
        ax3.tick_params(axis='both', labelsize=11, colors='#2c3e50', length=5, width=1)
        
        # Subtle spines
        for spine in ax3.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1.5)
        
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    # Full history table - IMPROVED
    st.markdown('<h3 class="section-header">Detailed Prediction Logs</h3>',
                unsafe_allow_html=True)
    
    # Sort history by timestamp descending for the table
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
        display_df = display_df.sort_values('timestamp', ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Date & Time",
                format="D MMM YYYY, h:mm a",
            ),
            "ram_size": "RAM",
            "storage_rom": "Storage",
            "processor": "Processor",
            "display_quality": "Display",
            "human_model_price": st.column_config.NumberColumn(
                "Predicted Price (Random Forest)",
                help="Price predicted by the Random Forest model",
                format="₹%,.2f",
            ),
            "ai_model_price": None,
        }
    )