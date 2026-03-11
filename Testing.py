import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Laptop Price Prediction System", layout="wide")

# Custom CSS to match the original Tkinter styling
st.markdown("""
    <style>
    /* Header block */
    .header-container {
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 30px;
    }
    .header-container h1 {
        color: white !important;
        margin: 0;
        font-family: 'Helvetica', sans-serif;
    }
    /* Custom Green Button */
    div.stButton > button:first-child {
        background-color: #27ae60 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        border: none !important;
        height: 50px;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #2ecc71 !important;
    }
    /* Results text */
    .results-text {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'dual_price_models.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Replace st.title with our custom styled header
st.markdown('<div class="header-container"><h1>Laptop Price Prediction System</h1></div>', unsafe_allow_html=True)

data = load_model()

if data is None:
    st.error("Model file 'dual_price_models.pkl' not found. Ensure the file is in the same folder as this script.")
    st.stop()

h_model = data['human_model']
a_model = data['ai_model']
le = data['label_encoder']
features = data['feature_names']

# Available options for drop-down menus
ram_options = ['8GB', '16GB', '32GB']
rom_options = ['256GB', '512GB', '1TB', '2TB']
display_options = ['FHD', '4K']
processor_options = list(le.classes_) if hasattr(le, 'classes_') else ['i3', 'i5', 'i7', 'i9', 'Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9']

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Select Specifications")
    ram = st.selectbox("RAM Size:", ram_options)
    rom = st.selectbox("Storage (ROM):", rom_options)
    processor = st.selectbox("Processor:", processor_options)
    display = st.selectbox("Display Quality:", display_options)
    
    predict_clicked = st.button("Get Prediction", type="primary", use_container_width=True)

def get_predictions(ram, rom, processor, display):
    # Prepare the input vector
    input_df = pd.DataFrame(0, index=[0], columns=features)

    # Map categorical features
    if f'ram_{ram}' in input_df.columns: input_df[f'ram_{ram}'] = 1
    if f'rom_{rom}' in input_df.columns: input_df[f'rom_{rom}'] = 1
    if f'display_resolution_{display}' in input_df.columns: input_df[f'display_resolution_{display}'] = 1
    
    # Handle Processor Encoding
    try:
        input_df['processor_encoded'] = le.transform([processor])[0]
    except:
        input_df['processor_encoded'] = 0

    # Get Predictions
    price_h = h_model.predict(input_df)[0]
    price_a = a_model.predict(input_df)[0]
    return price_h, price_a

with col2:
    if predict_clicked or 'predictions_made' not in st.session_state:
        st.session_state['predictions_made'] = True
        price_h, price_a = get_predictions(ram, rom, processor, display)
        
        # --- Save to Google Sheets (Silently runs if configured) ---
        if predict_clicked:
            if "gcp_service_account" in st.secrets:
                try:
                    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
                    client = gspread.authorize(creds)
                    
                    # Connects to your Google Sheet by its exact name
                    sheet = client.open("Laptop Price Predictions").sheet1
                    
                    # Log the inputs & predictions
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sheet.append_row([timestamp, ram, rom, processor, display, float(price_h), float(price_a)])
                except Exception as e:
                    st.warning(f"Google Sheets connection error: {e}")
        
        # Display nicely formatted result above chart
        st.markdown(f'<div class="results-text">Human-Implemented: {price_h:,.2f} &nbsp;|&nbsp; AI-Recommended: {price_a:,.2f}</div>', unsafe_allow_html=True)
        
        # --- Graphical Visualization ---
        plot_data = pd.DataFrame({
            'Model': ['Human Model', 'AI Model'],
            'Predicted Price': [price_h, price_a]
        })

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        sns.set_theme(style="whitegrid", palette="pastel")
        
        # Plot with appealing colors
        barplot = sns.barplot(x='Model', y='Predicted Price', hue='Model', data=plot_data, legend=False, ax=ax, palette=['#1A5F7A', '#DD5353'])
        
        ax.set_title(f'Price Prediction Comparison\n({ram}, {rom}, {processor}, {display})', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Estimated Price ($)', fontsize=14)
        ax.set_xlabel('Model Version', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        
        # Add price labels on top of bars
        for p in barplot.patches:
            height = p.get_height()
            ax.annotate(f'{height:,.2f}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'bottom', 
                            xytext = (0, 8), 
                            textcoords = 'offset points',
                            fontsize=12, fontweight='bold', color='#333333')

        fig.tight_layout()
        st.pyplot(fig)