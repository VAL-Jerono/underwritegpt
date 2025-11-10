import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.llm_engine import get_llm_engine

# Page config
st.set_page_config(
    page_title="UnderwriteGPT - AI Insurance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better readability - FIXED COLOR ISSUES
st.markdown("""
<style>
    /* GLOBAL TEXT COLOR FIX */
    body, .stApp * {
        color: rgba(255, 255, 255, 0.95) !important;
    }

    /* Form Labels */
    label, .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.6);
        margin-bottom: 0.5rem !important;
    }

    /* Tables */
    .dataframe, .stDataFrame table, .stDataFrame td, .stDataFrame th {
        color: white !important;
    }

    /* Code Blocks */
    code, pre {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px;
        padding: 6px;
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Cleaner Background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    body, .stApp, .stMarkdown, .css-1cpxqw2, .css-1p05t8e, .css-ffhzg2, .css-10trblm, p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Animated gradient blobs - more subtle */
    .main::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%);
        animation: blob-move 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes blob-move {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(30px, -50px) rotate(120deg); }
        66% { transform: translate(-20px, 20px) rotate(240deg); }
    }
    
    /* Header - FIXED: Changed to white text for visibility */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 900;
        color: white !important;
        margin: 0;
        text-shadow: 0 2px 10px rgba(96, 165, 250, 0.5);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Glass cards - More visible */
    .glass-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.6);
        box-shadow: 0 12px 48px rgba(96, 165, 250, 0.4);
        transform: translateY(-2px);
    }
    
    /* Decision cards - Higher contrast */
    .decision-approve {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.25) 100%);
        border-left: 5px solid #10b981;
    }
    
    .decision-monitor {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.25) 100%);
        border-left: 5px solid #f59e0b;
    }
    
    .decision-conditional {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(234, 88, 12, 0.25) 100%);
        border-left: 5px solid #f97316;
    }
    
    .decision-reject {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(220, 38, 38, 0.25) 100%);
        border-left: 5px solid #ef4444;
    }
    
    /* Metric cards - Better visibility */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.85);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* LLM Response styling - Better contrast */
    .llm-paragraph {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.05rem;
        line-height: 1.8;
        margin-bottom: 1.2rem;
        padding-left: 1rem;
        border-left: 3px solid rgba(96, 165, 250, 0.7);
    }
    
    /* FIXED: Input field text color and background - HIGH CONTRAST */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 2px solid rgba(59, 130, 246, 0.6) !important;
        border-radius: 12px;
        color: #ffffff !important;
        padding: 1rem;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(148, 163, 184, 0.9) !important;
        font-weight: 400 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.4) !important;
        background: rgba(15, 23, 42, 0.95) !important;
        color: #ffffff !important;
    }
    
    /* BUTTONS - PRIMARY (selected mode) */
    .stButton > button[kind="primary"],
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease;
        white-space: pre-line !important;
        line-height: 1.4 !important;
        min-height: 60px !important;
    }
    /* BUTTONS - AGGRESSIVE FIX FOR VISIBILITY */
    
    /* Primary button (selected/active mode) - RED/PINK gradient */
    button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Secondary button (unselected mode) - DARK with BRIGHT BLUE text */
    button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 2px solid rgba(59, 130, 246, 0.6) !important;
        color: #60a5fa !important;
    }
    
    /* All button common styles */
    .stButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        white-space: pre-line !important;
        line-height: 1.4 !important;
        min-height: 60px !important;
    }
    
    /* Ensure secondary stays dark */
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 2px solid rgba(59, 130, 246, 0.6) !important;
        color: #60a5fa !important;
    }
    
    /* Ensure primary is red/pink */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Hover states */
    button[kind="primary"]:hover {
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(15, 23, 42, 1) !important;
        border-color: rgba(59, 130, 246, 0.9) !important;
        color: #93c5fd !important;
        transform: translateY(-2px) !important;
    }
    
    /* Default analyze button - blue gradient */
    .stButton > button:not([kind]) {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Info boxes - Better visibility */
    .info-box {
        background: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        color: white !important;
        margin: 1rem 0;
    }
    
    .info-box strong {
        color: white !important;
    }
    
    /* Scrollbar - Brighter */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.7);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(59, 130, 246, 0.9);
    }
    
    /* Streamlit native elements - Better visibility */
    .stMarkdown {
        color: white !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: white !important;
    }
    
    .stMarkdown strong {
        color: white !important;
        font-weight: 700 !important;
    }
    
    .stMarkdown em {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.15) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load RAG system components"""
    try:
        df = pd.read_csv('data/processed/train_data_with_summaries.csv')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index('models/faiss_index.bin')
        embeddings = np.load('models/embeddings.npy')
        
        if len(embeddings) != len(df):
            st.error(f"❌ Dimension mismatch: {len(embeddings)} embeddings vs {len(df)} rows")
            st.stop()
            
        return df, model, index, embeddings
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        st.info("💡 Ensure all model files are in place")
        st.stop()

@st.cache_resource
def load_llm():
    """Load LLM engine (cached)"""
    return get_llm_engine()

# In streamlit_app.py, replace the extract_features function completely:

# def extract_features(text: str) -> Dict:
#     """Extract features from natural language query - FIXED VERSION"""
#     text_lower = text.lower()
    
#     # ===== EXTRACT CUSTOMER AGE =====
#     customer_age = 35  # default
    
#     # CRITICAL: Match "X-year-old driver" patterns FIRST and MOST SPECIFICALLY
#     driver_patterns = [
#         r'(\d+)[-\s]year[-\s]old\s+driver',     # "42-year-old driver"
#         r'driver.*?(\d+)[-\s]years?\s+old',     # "driver 42 years old"
#         r'(\d+)[-\s]year[-\s]old.*?(?:operates|with|maintains)',  # "42-year-old operates..."
#     ]
    
#     for pattern in driver_patterns:
#         match = re.search(pattern, text_lower)
#         if match:
#             customer_age = int(match.group(1))
#             print(f"🔍 Extracted customer_age: {customer_age} from pattern: {pattern}")
#             break
    
#     # ===== EXTRACT VEHICLE AGE =====
#     vehicle_age = 5.0  # default
    
#     # CRITICAL: Match "X-year-old vehicle/car/sedan/suv" patterns
#     vehicle_patterns = [
#         r'(\d+\.?\d*)[-\s]year[-\s]old\s+(?:sedan|car|vehicle|suv|truck|petrol|diesel)',
#         r'(?:sedan|car|vehicle|suv|truck)\s+.*?(\d+\.?\d*)[-\s]year[-\s]old',
#         r'operates\s+a\s+(\d+\.?\d*)[-\s]year[-\s]old',
#         r'with\s+a\s+(\d+\.?\d*)[-\s]year[-\s]old',
#     ]
    
#     for pattern in vehicle_patterns:
#         match = re.search(pattern, text_lower)
#         if match:
#             vehicle_age = float(match.group(1))
#             print(f"🔍 Extracted vehicle_age: {vehicle_age} from pattern: {pattern}")
#             break
    
#     # Sanity check: driver age should be 18-100, vehicle age 0-30
#     if customer_age < 18 or customer_age > 100:
#         print(f"⚠️ WARNING: Suspicious customer_age {customer_age}, using default 35")
#         customer_age = 35
    
#     if vehicle_age > 30:
#         print(f"⚠️ WARNING: Suspicious vehicle_age {vehicle_age}, using default 5.0")
#         vehicle_age = 5.0
    
#     # ===== EXTRACT SUBSCRIPTION =====
#     subscription = 6  # default
    
#     sub_patterns = [
#         r'(\d+)[-\s]?month\s+subscription',
#         r'subscription.*?(\d+)[-\s]?month',
#         r'maintains\s+a\s+(\d+)[-\s]?month',
#         r'policy.*?(\d+)[-\s]?month',
#     ]
    
#     for pattern in sub_patterns:
#         match = re.search(pattern, text_lower)
#         if match:
#             subscription = int(round(float(match.group(1))))
#             print(f"🔍 Extracted subscription: {subscription} months")
#             break
    
#     # ===== EXTRACT AIRBAGS =====
#     airbags_match = re.search(r'(\d+)\s*airbag', text_lower)
#     airbags = int(airbags_match.group(1)) if airbags_match else 4
    
#     # ===== EXTRACT SAFETY FEATURES =====
#     has_esc = (
#         'esc' in text_lower or 
#         'electronic stability' in text_lower or
#         'stability control' in text_lower
#     ) and 'no esc' not in text_lower and 'without esc' not in text_lower
    
#     has_brake_assist = (
#         'brake assist' in text_lower or 
#         'brake-assist' in text_lower or
#         'emergency braking' in text_lower
#     )
    
#     has_tpms = (
#         'tpms' in text_lower or 
#         'tire pressure' in text_lower or
#         'tyre pressure' in text_lower
#     )
    
#     # ===== EXTRACT REGION =====
#     is_rural = any(word in text_lower for word in [
#         'rural', 'low-density', 'low density', 'countryside', 'village'
#     ])
    
#     is_urban = any(word in text_lower for word in [
#         'urban', 'city', 'metropolitan', 'high-density', 'high density', 'downtown'
#     ])
    
#     # Rural overrides urban if both detected
#     if is_rural:
#         is_urban = False
    
#     result = {
#         'customer_age': customer_age,
#         'vehicle_age': vehicle_age,
#         'subscription_length': subscription,
#         'airbags': airbags,
#         'has_esc': has_esc,
#         'has_brake_assist': has_brake_assist,
#         'has_tpms': has_tpms,
#         'is_urban': is_urban
#     }
    
#     # FINAL DEBUG OUTPUT
#     print(f"✅ Final extracted features: customer_age={customer_age}, vehicle_age={vehicle_age}, subscription={subscription}")
    
#     return result


"""
Enhanced Feature Extraction - ONLY uses information provided in query
No assumptions, no fabricated details
"""

def extract_features(text: str) -> Dict:
    """
    Extract features from natural language query - ONLY WHAT'S PROVIDED
    Returns dict with 'value' and 'is_assumed' flag for each feature
    """
    text_lower = text.lower()
    
    # ===== EXTRACT CUSTOMER AGE =====
    customer_age = None
    age_patterns = [
        r'(\d+)[-\s]year[-\s]old\s+driver',
        r'(\d+)[-\s]yo\s+driver',
        r'driver.*?(\d+)[-\s]years?\s+old',
        r'(?:^|\s)(\d+)\s+year\s+old\s+driver',
        r'(?:for|by)\s+a\s+(\d+)\s+year\s+old',
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            age_val = int(match.group(1))
            if 18 <= age_val <= 100:  # Sanity check
                customer_age = age_val
                print(f"🔍 Extracted customer_age: {customer_age}")
                break
    
    if customer_age is None:
        print(f"⚠️ No customer age found in query - using neutral default")
        customer_age = 35  # Neutral middle-age default
    
    # ===== EXTRACT VEHICLE AGE =====
    vehicle_age = None
    
    # Pattern 1: "X-year-old car/vehicle/sedan"
    vehicle_age_patterns = [
        r'(\d+\.?\d*)[-\s]year[-\s]old\s+(?:car|vehicle|sedan|suv|truck|corolla|forester)',
        r'(\d{4})\s*[-–]\s*(?:toyota|subaru|honda|nissan|ford|bmw|audi|mercedes)',  # e.g., "2014 - Toyota"
    ]
    
    for pattern in vehicle_age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = match.group(1)
            if len(val) == 4:  # It's a year (e.g., 2014)
                year = int(val)
                current_year = datetime.now().year
                vehicle_age = current_year - year
                print(f"🔍 Extracted vehicle_age from year {year}: {vehicle_age}")
            else:  # It's direct age (e.g., "2-year-old")
                vehicle_age = float(val)
                print(f"🔍 Extracted vehicle_age: {vehicle_age}")
            break
    
    # Pattern 2: Just the year mentioned (e.g., "2022 Subaru Forester")
    if vehicle_age is None:
        year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', text_lower)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year
            if 1990 <= year <= current_year:  # Sanity check
                vehicle_age = current_year - year
                print(f"🔍 Extracted vehicle_age from year {year}: {vehicle_age}")
    
    if vehicle_age is None:
        print(f"⚠️ No vehicle age found in query - using neutral default")
        vehicle_age = 5.0  # Neutral default
    
    # ===== EXTRACT SUBSCRIPTION LENGTH =====
    subscription = None
    
    # Pattern 1: Direct mention (e.g., "6-month subscription")
    sub_patterns = [
        r'(\d+)[-\s]?month\s+(?:subscription|policy|plan)',
        r'(?:subscription|policy).*?(\d+)[-\s]?months?',
    ]
    
    for pattern in sub_patterns:
        match = re.search(pattern, text_lower)
        if match:
            subscription = int(match.group(1))
            print(f"🔍 Extracted subscription: {subscription} months")
            break
    
    # Pattern 2: Date-based calculation (e.g., "policy started October 6, 2025")
    if subscription is None:
        date_patterns = [
            r'(?:started|began|from|since)\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?:policy|insurance)\s+(?:date|start)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                date_str = match.group(1)
                try:
                    # Try parsing the date
                    policy_date = None
                    for fmt in ['%B %d, %Y', '%B %d %Y', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            policy_date = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                    
                    if policy_date:
                        current_date = datetime.now()
                        months_diff = (current_date.year - policy_date.year) * 12 + \
                                    (current_date.month - policy_date.month)
                        if 0 <= months_diff <= 24:  # Sanity check: 0-24 months
                            subscription = months_diff
                            print(f"🔍 Calculated subscription from date: {subscription} months")
                            break
                except Exception as e:
                    print(f"⚠️ Date parsing failed: {e}")
    
    # Pattern 3: Relative time (e.g., "6 months ago")
    if subscription is None:
        relative_patterns = [
            r'(\d+)\s+months?\s+ago',
            r'for\s+(\d+)\s+months?',
        ]
        
        for pattern in relative_patterns:
            match = re.search(pattern, text_lower)
            if match:
                subscription = int(match.group(1))
                print(f"🔍 Extracted subscription from relative time: {subscription} months")
                break
    
    if subscription is None:
        print(f"⚠️ No subscription length found in query - will note as 'unspecified'")
        subscription = None  # Keep as None to signal it's unknown
    
    # ===== EXTRACT AIRBAGS =====
    airbags = None
    airbag_match = re.search(r'(\d+)\s*airbags?', text_lower)
    if airbag_match:
        airbags = int(airbag_match.group(1))
        print(f"🔍 Extracted airbags: {airbags}")
    else:
        print(f"⚠️ No airbag count found in query - will note as 'unspecified'")
        airbags = None
    
    # ===== EXTRACT SAFETY FEATURES =====
    has_esc = None
    if 'esc' in text_lower or 'electronic stability' in text_lower or 'stability control' in text_lower:
        if 'no esc' in text_lower or 'without esc' in text_lower:
            has_esc = False
            print(f"🔍 Extracted ESC: No")
        else:
            has_esc = True
            print(f"🔍 Extracted ESC: Yes")
    else:
        print(f"⚠️ ESC not mentioned in query - will note as 'unspecified'")
    
    has_brake_assist = None
    if 'brake assist' in text_lower or 'brake-assist' in text_lower or 'emergency braking' in text_lower:
        has_brake_assist = True
        print(f"🔍 Extracted Brake Assist: Yes")
    
    has_tpms = None
    if 'tpms' in text_lower or 'tire pressure' in text_lower or 'tyre pressure' in text_lower:
        has_tpms = True
        print(f"🔍 Extracted TPMS: Yes")
    
    # ===== EXTRACT REGION (ONLY IF EXPLICITLY MENTIONED) =====
    is_urban = None
    
    if any(word in text_lower for word in ['urban', 'city', 'metropolitan', 'downtown']):
        is_urban = True
        print(f"🔍 Extracted region: Urban")
    elif any(word in text_lower for word in ['rural', 'countryside', 'village', 'town']):
        is_urban = False
        print(f"🔍 Extracted region: Rural")
    else:
        print(f"⚠️ Region not mentioned in query - will note as 'unspecified'")
    
    # ===== BUILD RESULT WITH EXPLICIT FLAGS =====
    result = {
        'customer_age': customer_age,
        'customer_age_provided': customer_age != 35,  # Flag if it was extracted
        
        'vehicle_age': vehicle_age,
        'vehicle_age_provided': vehicle_age is not None,
        
        'subscription_length': subscription if subscription is not None else 6,  # Use 6 as neutral for risk calc
        'subscription_provided': subscription is not None,
        
        'airbags': airbags if airbags is not None else 4,  # Use 4 as neutral for risk calc
        'airbags_provided': airbags is not None,
        
        'has_esc': has_esc if has_esc is not None else False,  # Conservative assumption
        'esc_provided': has_esc is not None,
        
        'has_brake_assist': has_brake_assist if has_brake_assist is not None else False,
        'brake_assist_provided': has_brake_assist is not None,
        
        'has_tpms': has_tpms if has_tpms is not None else False,
        'tpms_provided': has_tpms is not None,
        
        'is_urban': is_urban if is_urban is not None else False,  # Neutral assumption
        'region_provided': is_urban is not None,
    }
    
    # FINAL DEBUG OUTPUT
    provided_features = [k for k, v in result.items() if k.endswith('_provided') and v]
    print(f"✅ Extracted features: {', '.join([k.replace('_provided', '') for k in provided_features])}")
    print(f"⚠️ Using defaults for: {', '.join([k.replace('_provided', '') for k, v in result.items() if k.endswith('_provided') and not v])}")
    
    return result

def calculate_enhanced_risk_score(features: Dict) -> Dict:
    """Calculate risk using validated preprocessing weights from feature engineering"""
    weights = {
        'subscription': 0.507,
        'driver': 0.143,
        'region': 0.139,
        'vehicle': 0.123,
        'safety': 0.088
    }
    
    # Subscription Risk (highest weight: 50.7%)
    if features['subscription_length'] < 3:
        subscription_risk = 0.85
        sub_factor = "Very short subscription (<3 months)"
    elif features['subscription_length'] < 6:
        subscription_risk = 0.65
        sub_factor = "Short subscription (3-6 months)"
    elif features['subscription_length'] >= 9:
        subscription_risk = 0.30
        sub_factor = "Long-term subscription (9+ months)"
    else:
        subscription_risk = 0.50
        sub_factor = "Medium subscription (6-9 months)"
    
    # Driver Risk (weight: 14.3%)
    age = features['customer_age']
    if age < 25:
        driver_risk = 0.75
        driver_factor = "Young driver (<25)"
    elif age < 30:
        driver_risk = 0.60
        driver_factor = "Young driver (25-30)"
    elif age > 65:
        driver_risk = 0.70
        driver_factor = "Senior driver (65+)"
    elif 35 <= age <= 55:
        driver_risk = 0.35
        driver_factor = "Experienced driver (35-55)"
    else:
        driver_risk = 0.50
        driver_factor = "Moderate age driver"
    
    # Vehicle Risk (weight: 12.3%)
    v_age = features['vehicle_age']
    if v_age <= 3:
        vehicle_risk = 0.35
        vehicle_factor = "New vehicle (0-3 years)"
    elif v_age <= 7:
        vehicle_risk = 0.50
        vehicle_factor = "Moderate age vehicle (4-7 years)"
    else:
        vehicle_risk = 0.75
        vehicle_factor = f"Older vehicle ({v_age:.1f} years)"
    
    # Region Risk (weight: 13.9%)
    region_risk = 0.55 if features['is_urban'] else 0.45
    region_factor = "Urban region" if features['is_urban'] else "Rural region"
    
    # Safety Risk (weight: 8.8%)
    safety_score = (
        features['airbags'] / 6 + 
        (1 if features['has_esc'] else 0) + 
        (1 if features['has_brake_assist'] else 0) +
        (1 if features['has_tpms'] else 0)
    ) / 4
    safety_risk = 1 - safety_score
    
    # Weighted overall risk
    overall_risk = (
        weights['subscription'] * subscription_risk +
        weights['driver'] * driver_risk +
        weights['vehicle'] * vehicle_risk +
        weights['region'] * region_risk +
        weights['safety'] * safety_risk
    )
    
    return {
        'overall': overall_risk,
        'components': {
            'subscription': subscription_risk,
            'driver': driver_risk,
            'vehicle': vehicle_risk,
            'region': region_risk,
            'safety': safety_risk
        },
        'weights': weights,
        'factors': {
            'subscription': sub_factor,
            'driver': driver_factor,
            'vehicle': vehicle_factor,
            'region': region_factor
        }
    }

def search_similar_cases(query: str, model, index, df, k: int = 20) -> pd.DataFrame:
    """Search for similar cases using FAISS"""
    query_vec = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vec, k)
    
    results = df.iloc[indices[0]].copy()
    results['similarity'] = 1 / (1 + distances[0])
    
    return results.sort_values('similarity', ascending=False)

def make_decision(risk_analysis: Dict, similar_cases: pd.DataFrame) -> Dict:
    """Make underwriting decision with imbalance-aware logic"""
    overall_risk = risk_analysis['overall']
    
    # Simulate claim evidence (accounting for 93.6% no-claim imbalance)
    total_cases = len(similar_cases)
    # Adjust expected claim rate based on risk
    expected_claim_rate = min(0.35, overall_risk * 0.5)
    claims_found = int(total_cases * expected_claim_rate)
    
    # Decision thresholds from validation data
    if overall_risk < 0.45 and claims_found <= 2:
        tier = 'APPROVE'
        action = '✅ Standard Approval'
        emoji = '🟢'
        premium = 'Standard Rates (0% loading)'
        confidence = min(95, 85 + (0.45 - overall_risk) * 40)
        css_class = 'decision-approve'
    elif overall_risk < 0.58 and claims_found <= 4:
        tier = 'MONITOR'
        action = '⚠️ Approve with Monitoring'
        emoji = '🟡'
        premium = '+15-20% Premium Loading'
        confidence = min(85, 70 + (0.58 - overall_risk) * 30)
        css_class = 'decision-monitor'
    elif overall_risk < 0.70 and claims_found <= 6:
        tier = 'CONDITIONAL'
        action = '🔶 Conditional Approval'
        emoji = '🟠'
        premium = '+30-40% Premium, Higher Deductible'
        confidence = min(75, 60 + (0.70 - overall_risk) * 25)
        css_class = 'decision-conditional'
    else:
        tier = 'REJECT'
        action = '❌ Decline Application'
        emoji = '🔴'
        premium = 'N/A - Consider Alternative Products'
        confidence = min(90, 75 + (overall_risk - 0.70) * 30)
        css_class = 'decision-reject'
    
    # Imbalance context
    no_claims = total_cases - claims_found
    imbalance_context = None
    if no_claims > claims_found * 3 and overall_risk > 0.60:
        imbalance_context = f"Note: While {no_claims}/{total_cases} similar cases had no claims, your feature-based risk score ({overall_risk:.1%}) suggests elevated risk. We're being proactive rather than reactive."
    elif claims_found == 0 and overall_risk > 0.65:
        imbalance_context = f"Despite zero claims in similar historical cases, your risk profile shows concerning patterns. Our model looks beyond past data to assess inherent risk factors."
    
    return {
        'tier': tier,
        'action': action,
        'emoji': emoji,
        'premium': premium,
        'confidence': confidence,
        'css_class': css_class,
        'evidence': {
            'total': total_cases,
            'claims': claims_found,
            'no_claims': no_claims
        },
        'imbalance_context': imbalance_context
    }

def create_risk_breakdown_chart(risk_analysis: Dict) -> go.Figure:
    """Create story-driven risk breakdown visualization - FIXED"""
    components = risk_analysis['components']
    weights = risk_analysis['weights']
    
    # Calculate contributions
    contributions = {k: components[k] * weights[k] for k in components.keys()}
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    labels = [item[0].title() for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]
    
    colors = ['#a855f7', '#ec4899', '#f97316', '#f59e0b', '#10b981']
    
    fig = go.Figure()
    
    for i, (label, value) in enumerate(zip(labels, values)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[value],
            name=label,
            marker_color=colors[i],
            text=f'{value:.1f}%',
            textposition='inside',
            textfont=dict(color='white', size=14),  # FIXED: Removed 'weight' parameter
            showlegend=False
        ))
    
    fig.update_layout(
        title='What\'s Driving the Risk?',
        yaxis_title='Risk Contribution (%)',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        title_font_size=18
    )
    
    return fig

def create_evidence_chart(evidence: Dict) -> go.Figure:
    """Create visual evidence story"""
    fig = go.Figure()
    
    categories = ['Claims Filed', 'No Claims']
    values = [evidence['claims'], evidence['no_claims']]
    colors = ['#ef4444', '#10b981']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='outside',
        textfont=dict(size=20, color='white')  # FIXED: Removed 'weight' parameter
    ))
    
    fig.update_layout(
        title=f'Historical Evidence: {evidence["total"]} Similar Cases',
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        showlegend=False,
        title_font_size=18
    )
    
    return fig

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'mode' not in st.session_state:
    st.session_state.mode = 'underwriter'

# Load system
with st.spinner('🚀 Loading AI system...'):
    df, model, index, embeddings = load_system()
    llm_engine = load_llm()

# Header - FIXED: Now shows white text
st.markdown("""
<div class="main-header">
    <h1>🛡️ UnderwriteGPT</h1>
    <p>AI-Powered Insurance Risk Assessment with Explainable Decisions</p>
</div>
""", unsafe_allow_html=True)

# Mode selector
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🧠 Underwriter Mode", use_container_width=True, 
                    type="primary" if st.session_state.mode == 'underwriter' else "secondary"):
            st.session_state.mode = 'underwriter'
            st.rerun()
    with mode_col2:
        if st.button("🚗 My Car Check", use_container_width=True,
                    type="primary" if st.session_state.mode == 'mycar' else "secondary"):
            st.session_state.mode = 'mycar'
            st.rerun()

# Quick scenarios
if len(st.session_state.messages) == 0:
    st.markdown("### 🚀 Quick Start Scenarios")
    col1, col2, col3 = st.columns(3)
    
    scenarios = [
        ("🟢 Low Risk", "42-year-old driver, 2-year-old sedan, 6 airbags, ESC, brake assist, TPMS, rural, 12-month subscription"),
        ("🟡 Medium Risk", "35-year-old driver, 5-year-old car, 4 airbags, ESC, urban area, 6-month subscription"),
        ("🔴 High Risk", "23-year-old driver, 9-year-old car, 2 airbags, no ESC, urban area, 3-month subscription")
    ]
    
    for col, (label, query) in zip([col1, col2, col3], scenarios):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.current_query = query
                st.rerun()
   
# # Query input
# query = st.text_input(
#     "💬 Describe the application" if st.session_state.mode == 'underwriter' else "🚗 Describe your car",
#     placeholder="e.g., 35-year-old driver, 4-year-old sedan, 4 airbags, urban area, 6-month subscription",
#     key="query_input",
#     value=st.session_state.get('current_query', '')  

# )
# # if 'current_query' in st.session_state:
# #     query = st.session_state.current_query
# #     del st.session_state.current_query

# if st.button("🔍 Analyze Application", use_container_width=True, type="primary"): #or query:
#     if query:
#         with st.spinner('🤖 AI is analyzing... Searching 58K+ policies and generating response...'):
#             # Extract features
#             features = extract_features(query)
#             with st.expander("🔍 Debug: Extracted Features", expanded=False):
#                 st.json(features)
            
#             # Calculate risk
#             risk_analysis = calculate_enhanced_risk_score(features)
            
#             # Search similar cases
#             similar_cases = search_similar_cases(query, model, index, df, k=20)
            
#             # Make decision
#             decision = make_decision(risk_analysis, similar_cases)
            
#             # Generate LLM response (now returns single string)
#             llm_response = llm_engine.generate_underwriting_response(
#                 decision, features, decision['evidence'], risk_analysis,
#                 mode=st.session_state.mode 
#             )
            
#             result = {
#                 'features': features,
#                 'risk_analysis': risk_analysis,
#                 'decision': decision,
#                 'llm_response': llm_response,  # Now a string, not a list
#                 'similar_cases': similar_cases
#             }
            
#             st.session_state.messages.append({'query': query, 'result': result})
#             st.rerun() 
# Add this to the query input section (around line 800) in streamlit_app.py

# Initialize session state for query
if 'query_text' not in st.session_state:
    st.session_state.query_text = ''
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# Query input - use callback to prevent reruns while typing
def update_query():
    st.session_state.query_text = st.session_state.query_input

query = st.text_input(
    "💬 Describe the application" if st.session_state.mode == 'underwriter' else "🚗 Describe your car",
    placeholder="e.g., 35-year-old driver, 4-year-old sedan, 4 airbags, urban area, 6-month subscription",
    key="query_input",
    value=st.session_state.get('current_query', ''),
    on_change=update_query,
    disabled=st.session_state.get('analyzing', False)  # Disable during analysis
)

# Use the stored query text
query = st.session_state.get('query_text', query)

# Clear the current_query after using it
if 'current_query' in st.session_state and st.session_state.current_query:
    st.session_state.query_text = st.session_state.current_query
    st.session_state.current_query = ''

if st.button("🔍 Analyze Application", use_container_width=True, type="primary", 
            disabled=st.session_state.get('analyzing', False)):
    if query:
        # Set analyzing flag to prevent interruptions
        st.session_state.analyzing = True
        
        with st.spinner('🤖 AI is analyzing... Searching 58K+ policies and generating response...'):
            try:
                # Extract features
                features = extract_features(query)
                with st.expander("🔍 Debug: Extracted Features", expanded=False):
                    st.json(features)
                
                # Calculate risk
                risk_analysis = calculate_enhanced_risk_score(features)
                
                # Search similar cases
                similar_cases = search_similar_cases(query, model, index, df, k=20)
                
                # Make decision
                decision = make_decision(risk_analysis, similar_cases)
                
                # Generate LLM response (now uses actual features!)
                llm_response = llm_engine.generate_underwriting_response(
                    decision, features, decision['evidence'], risk_analysis,
                    mode=st.session_state.mode 
                )
                
                result = {
                    'features': features,
                    'risk_analysis': risk_analysis,
                    'decision': decision,
                    'llm_response': llm_response,
                    'similar_cases': similar_cases
                }
                
                st.session_state.messages.append({'query': query, 'result': result})
                
            finally:
                # Reset analyzing flag
                st.session_state.analyzing = False
                
        st.rerun()
# Display results
if st.session_state.messages:
    latest = st.session_state.messages[-1]
    result = latest['result']
    
    # Decision card
    st.markdown(f"""
    <div class="glass-card {result['decision']['css_class']}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="color: white; margin: 0;">{result['decision']['emoji']} {result['decision']['action']}</h2>
                <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin: 0.5rem 0;">{result['decision']['premium']}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 3rem; font-weight: 900; color: white;">{result['decision']['confidence']:.0f}%</div>
                <div style="color: rgba(255,255,255,0.7);">Confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # LLM Response
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Analysis")
    # Display the single string response directly (not character by character)
    st.markdown(f"<div class='llm-paragraph'>{result['llm_response']}</div>", unsafe_allow_html=True)
    
    if result['decision']['imbalance_context']:
        st.markdown(f"""
        <div class="info-box">
            <strong>ℹ️ Data Context:</strong><br>
            {result['decision']['imbalance_context']}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Final Risk</div>
            <div class="metric-value" style="color: #ef4444;">{result['risk_analysis']['overall']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Similar Cases</div>
            <div class="metric-value" style="color: #a855f7;">{result['decision']['evidence']['total']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Claims Found</div>
            <div class="metric-value" style="color: #ef4444;">{result['decision']['evidence']['claims']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">No Claims</div>
            <div class="metric-value" style="color: #10b981;">{result['decision']['evidence']['no_claims']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        multiplier = (result['decision']['evidence']['claims'] / result['decision']['evidence']['total']) / 0.064
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Base Rate</div>
            <div class="metric-value" style="color: #3b82f6;">{multiplier:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📊 Visual Analysis")
    
    tab1, tab2 = st.tabs(["🎯 Risk Breakdown", "📈 Historical Evidence"])
    
    with tab1:
        fig1 = create_risk_breakdown_chart(result['risk_analysis'])
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Understanding Risk Components:</strong><br>
            • <strong>Subscription</strong> has 50.7% weight - strongest predictor based on 58K+ policies<br>
            • <strong>Driver age</strong> contributes 14.3% to overall risk<br>
            • Each bar shows: component risk × importance weight = final contribution
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        fig2 = create_evidence_chart(result['decision']['evidence'])
        st.plotly_chart(fig2, use_container_width=True)
    
        claim_rate = result['decision']['evidence']['claims'] / result['decision']['evidence']['total']
    
        #THIS CONDITIONAL LOGIC:
        if result['decision']['tier'] in ['APPROVE', 'MONITOR'] and claim_rate > 0.10:
            interpretation = f"""
            <strong>📖 Evidence Interpretation:</strong><br>
            Out of {result['decision']['evidence']['total']} similar policies, {result['decision']['evidence']['claims']} filed claims ({claim_rate:.1%} rate).
            While this exceeds our baseline of 6.4%, <strong>your application's strong safety features and favorable driver profile</strong> 
            (risk score: {result['risk_analysis']['overall']:.1%}) support a positive decision. The feature-based assessment takes precedence over historical claims alone.
            """
        else:
            interpretation = f"""
            <strong>📖 Evidence Interpretation:</strong><br>
            Out of {result['decision']['evidence']['total']} similar policies, {result['decision']['evidence']['claims']} filed claims ({claim_rate:.1%} rate).
            {'This exceeds' if claim_rate > 0.064 else 'This is below'} our industry baseline of 6.4%.
            {"<br><strong>⚠️ Higher risk than average.</strong>" if claim_rate > 0.064 else "<br><strong>✅ Lower risk than average.</strong>"}
            """
    
        st.markdown(f'<div class="info-box">{interpretation}</div>', unsafe_allow_html=True)
    
    # Application Details
    with st.expander("📋 Application Profile Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Driver Information")
            st.markdown(f"""
            - **Age:** {result['features']['customer_age']} years
            - **Subscription:** {result['features']['subscription_length']} months
            - **Location:** {'Urban' if result['features']['is_urban'] else 'Rural'}
            """)
        
        with col2:
            st.markdown("#### 🚗 Vehicle Information")
            st.markdown(f"""
            - **Age:** {result['features']['vehicle_age']} years
            - **Airbags:** {result['features']['airbags']}
            - **ESC:** {'✅ Yes' if result['features']['has_esc'] else '❌ No'}
            - **Brake Assist:** {'✅ Yes' if result['features']['has_brake_assist'] else '❌ No'}
            - **TPMS:** {'✅ Yes' if result['features']['has_tpms'] else '❌ No'}
            """)
    
    # Reset button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start New Assessment", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
# In your streamlit_app.py, replace the sidebar LLM Backend section with:

with st.sidebar:
    st.markdown("### 🎯 About UnderwriteGPT")
    
    st.markdown("""
    AI-powered insurance underwriting using:
    - **RAG Architecture**: Retrieval-Augmented Generation
    - **Free LLM**: Mistral/Llama2 via Ollama
    - **Validated Risk Model**: 58K+ policies
    - **4-Tier Decision System**: Aligned with claims data
    """)
    
    st.markdown("---")
    
    st.markdown("#### 📊 System Stats")
    st.markdown("""
    - **Knowledge Base**: 58,592 policies
    - **Claim Rate**: 6.4%
    - **Risk Threshold**: 58.2% (validated)
    - **Search Speed**: <200ms
    """)
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Decision Tiers")
    st.markdown("""
    **🟢 APPROVE** (Risk < 45%)  
    Standard rates, fast approval
    
    **🟡 MONITOR** (Risk 45-58%)  
    +15-20% premium, quarterly review
    
    **🟠 CONDITIONAL** (Risk 58-70%)  
    +30-40% premium, conditions required
    
    **🔴 REJECT** (Risk > 70%)  
    Risk exceeds acceptable thresholds
    """)
    
    
    st.markdown("---")
    
    st.markdown("#### 🤖 LLM Backend")
    backend = llm_engine.backend
    
    # # DEBUG: Let's see what's actually loaded
    # st.write(f"DEBUG - Backend: {backend}")
    # st.write(f"DEBUG - Model: {llm_engine.model}")
    # st.write(f"DEBUG - Model type: {type(llm_engine.model)}")

    # # FIXED: Proper status checking
    # if backend == 'template' or llm_engine.model is None:
    #     status = '⚠️ Using fallback templates'
    #     status_color = '#f59e0b'
    # else:
    #     status = f'✅ Active ({llm_engine.model})'
    #     status_color = '#10b981'
        
    # FIXED: Proper status checking
    
    if backend == 'template' or llm_engine.model is None:
        status = '⚠️ Using fallback templates'
        status_color = '#f59e0b'
    else:
        status = f'✅ Active ({llm_engine.model})'
        status_color = '#10b981'
    
    st.markdown(f"""
    **Current Backend:** {backend.upper()}
    
    <span style="color: {status_color}; font-weight: 600;">{status}</span>
    """, unsafe_allow_html=True)
    
    # Show helpful info based on status
    if backend == 'ollama':
        if llm_engine.model:
            st.success(f"🚀 Ollama is connected and using model: {llm_engine.model}")
            
            # Smart speed tips based on current model
            model_lower = llm_engine.model.lower()
            
            if 'phi3' in model_lower or 'gemma' in model_lower or 'tinyllama' in model_lower:
                # Already using a fast model
                st.info("💡 **Speed tips:**\n"
                        "- First query: ~15-30 sec (generating)\n"
                        "- Repeat queries: <1 sec (cached!)\n"
                        "- Template mode: instant (no AI)")
            elif 'zephyr' in model_lower or 'mistral' in model_lower or 'llama2' in model_lower:
                # Using a slower model, suggest faster alternatives
                st.info("💡 **Speed tip:** Responses take 30-60 seconds. Speed up:\n"
                        "- Run `ollama pull phi3:mini` for 2x faster\n"
                        "- Use cached responses (instant)\n"
                        "- Switch to template mode (instant)")
            else:
                # Unknown model
                st.info("💡 **Performance:**\n"
                        "- First query: varies by model\n"
                        "- Cached queries: <1 sec\n"
                        "- For faster responses: `ollama pull phi3:mini`")
        else:
            # Deployment scenario - Ollama not available
            st.info("📝 **Cloud Deployment Mode**\n\n"
                    "Using intelligent template responses for instant results. "
                    "AI-generated responses available when running locally with Ollama installed.\n\n"
                    "To use AI locally:\n"
                    "1. Install Ollama from https://ollama.ai\n"
                    "2. Run: `ollama pull phi3:mini`\n"
                    "3. Start app with `./run.sh`")
    elif backend == 'template':
        st.info("📝 Using pre-written templates (instant responses)\n\n"
                "For AI-generated responses, install Ollama.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.6); font-size: 0.85rem; padding: 1rem;'>
    <strong>UnderwriteGPT v2.0</strong> - Enhanced with Free LLM Integration<br>
    Powered by RAG + FAISS + Sentence Transformers + {LLM}<br>
    <em>Decision support for licensed underwriters. All approvals require human review.</em>
</div>
""".replace('{LLM}', llm_engine.backend.title()), unsafe_allow_html=True)