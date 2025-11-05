import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re

# Page config
st.set_page_config(
    page_title="UnderwriteGPT Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with chat interface
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Bot message */
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 85%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Risk badges - compact */
    .risk-badge-high {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-badge-medium-high {
        background: linear-gradient(135deg, #ff9933 0%, #ff6600 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-badge-medium {
        background: linear-gradient(135deg, #ffcc00 0%, #ff9900 100%);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #44cc44 0%, #339933 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Action items */
    .action-item {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    /* Evidence cards */
    .evidence-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .evidence-header {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    /* Quick start buttons */
    .quick-button {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s;
        display: inline-block;
    }
    
    .quick-button:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load all resources"""
    df = pd.read_csv('data/processed/train_data_with_summaries.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        # Try loading existing indices
        index_claims = faiss.read_index('models/faiss_claims_index.bin')
        index_no_claims = faiss.read_index('models/faiss_no_claims_index.bin')
        
        # Load embeddings to get the correct dataframes
        embeddings = np.load('models/embeddings.npy')
        
        # Check if dimensions match
        if len(embeddings) != len(df):
            st.warning(f"⚠️ Embeddings ({len(embeddings)}) don't match data ({len(df)}). Rebuilding indices...")
            raise ValueError("Dimension mismatch")
        
        claim_mask = df['claim_status'] == 1
        df_claims = df[claim_mask].reset_index(drop=True)
        df_no_claims = df[~claim_mask].reset_index(drop=True)
        
    except Exception as e:
        # Rebuild everything from scratch
        st.info("🔄 Building search indices (first time only, ~30 seconds)...")
        
        embeddings = np.load('models/embeddings.npy')
        
        # Verify dimensions
        if len(embeddings) != len(df):
            st.error(f"""
            ❌ **Dimension Mismatch Error**
            
            - Data file: {len(df)} rows
            - Embeddings file: {len(embeddings)} rows
            
            **Fix:** Re-run the embedding generation notebook (03_rag_retrieval.ipynb) 
            to regenerate embeddings.npy that matches your current data file.
            """)
            st.stop()
        
        claim_mask = df['claim_status'] == 1
        
        embeddings_claims = embeddings[claim_mask]
        embeddings_no_claims = embeddings[~claim_mask]
        
        dimension = embeddings.shape[1]
        
        # Build indices
        index_claims = faiss.IndexFlatL2(dimension)
        index_claims.add(embeddings_claims)
        
        index_no_claims = faiss.IndexFlatL2(dimension)
        index_no_claims.add(embeddings_no_claims)
        
        # Save indices
        faiss.write_index(index_claims, 'models/faiss_claims_index.bin')
        faiss.write_index(index_no_claims, 'models/faiss_no_claims_index.bin')
        
        df_claims = df[claim_mask].reset_index(drop=True)
        df_no_claims = df[~claim_mask].reset_index(drop=True)
        
        st.success("✅ Indices built successfully!")
    
    return df, model, index_claims, index_no_claims, df_claims, df_no_claims

def extract_features(text):
    """Extract key features from query"""
    text_lower = text.lower()
    
    all_ages = re.findall(r'(\d+)[-\s]?year[-\s]?old', text_lower)
    
    if len(all_ages) >= 2:
        age = int(all_ages[0])
        v_age = float(all_ages[1])
    elif len(all_ages) == 1:
        age = int(all_ages[0])
        v_age = 5.0
    else:
        age = 35
        v_age = 5.0
    
    airbags_match = re.search(r'(\d+)\s*airbag', text_lower)
    airbags = int(airbags_match.group(1)) if airbags_match else 4
    
    has_esc = bool(re.search(r'\besc\b', text_lower))
    has_no_esc = bool(re.search(r'no\s+esc', text_lower))
    if has_no_esc:
        has_esc = False
    
    has_brake_assist = 'brake assist' in text_lower
    
    return {
        'age': age, 
        'vehicle_age': v_age, 
        'airbags': airbags, 
        'has_esc': has_esc,
        'has_brake_assist': has_brake_assist
    }

def calculate_feature_risk(features, base_rate):
    """Calculate risk from features"""
    risk_adjustment = 0.0
    
    if features['age'] < 25:
        risk_adjustment += 0.025
    elif features['age'] < 30:
        risk_adjustment += 0.012
    elif features['age'] > 65:
        risk_adjustment += 0.018
    elif features['age'] > 60:
        risk_adjustment += 0.008
    else:
        risk_adjustment -= 0.005
    
    if features['vehicle_age'] > 10:
        risk_adjustment += 0.035
    elif features['vehicle_age'] > 7:
        risk_adjustment += 0.020
    elif features['vehicle_age'] > 5:
        risk_adjustment += 0.010
    elif features['vehicle_age'] <= 3:
        risk_adjustment -= 0.015
    
    if features['airbags'] <= 2:
        risk_adjustment += 0.015
    elif features['airbags'] >= 6:
        risk_adjustment -= 0.010
    
    if not features['has_esc']:
        risk_adjustment += 0.012
    if not features['has_brake_assist']:
        risk_adjustment += 0.008
    
    feature_risk = base_rate + risk_adjustment
    feature_risk = max(0.01, min(0.30, feature_risk))
    
    return feature_risk

def search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5):
    """Search both indices"""
    query_vec = model.encode([query])
    
    dist_c, idx_c = idx_claims.search(query_vec, k)
    results_c = df_claims.iloc[idx_c[0]].copy()
    results_c['distance'] = dist_c[0]
    
    dist_nc, idx_nc = idx_no_claims.search(query_vec, k)
    results_nc = df_no_claims.iloc[idx_nc[0]].copy()
    results_nc['distance'] = dist_nc[0]
    
    combined = pd.concat([results_c, results_nc]).reset_index(drop=True)
    combined['weight'] = 1.0 / (1.0 + combined['distance'])
    combined = combined.sort_values('distance').reset_index(drop=True)
    
    return combined

def calculate_weighted_risk(cases, base_rate):
    """Calculate weighted risk"""
    weighted_sum = (cases['claim_status'] * cases['weight']).sum()
    total_weight = cases['weight'].sum()
    
    if total_weight == 0:
        return base_rate
    
    weighted_risk = weighted_sum / total_weight
    deviation = weighted_risk - 0.5
    adjusted_risk = base_rate + (deviation * base_rate * 2)
    adjusted_risk = max(0.01, min(0.25, adjusted_risk))
    
    return adjusted_risk

def determine_risk_level(rag_risk, feature_risk, base_rate):
    """Determine final risk level"""
    combined = (0.70 * feature_risk) + (0.30 * rag_risk)
    
    if combined >= 0.12:
        return "HIGH RISK", "risk-badge-high", combined, "🔴"
    elif combined >= 0.09:
        return "MEDIUM-HIGH", "risk-badge-medium-high", combined, "🟠"
    elif combined >= 0.07:
        return "MEDIUM", "risk-badge-medium", combined, "🟡"
    else:
        return "LOW RISK", "risk-badge-low", combined, "🟢"

def create_risk_gauge(risk_score):
    """Create compact risk gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 16}},
        number = {'suffix': "%", 'font': {'size': 32}},
        gauge = {
            'axis': {'range': [0, 25], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 7], 'color': '#d4edda'},
                {'range': [7, 9], 'color': '#fff3cd'},
                {'range': [9, 12], 'color': '#ffe5cc'},
                {'range': [12, 25], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 12}
    )
    
    return fig

def create_evidence_chart(cases):
    """Create evidence distribution chart"""
    claim_counts = cases.head(10)['claim_status'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Claims', 'No Claims'],
            y=[claim_counts.get(1, 0), claim_counts.get(0, 0)],
            marker_color=['#ff4444', '#44cc44'],
            text=[claim_counts.get(1, 0), claim_counts.get(0, 0)],
            textposition='auto',
            textfont={'size': 16, 'color': 'white'}
        )
    ])
    
    fig.update_layout(
        title="Similar Cases Found",
        xaxis_title="",
        yaxis_title="Count",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def create_comparison_radar(features, avg_features):
    """Create feature comparison radar"""
    categories = ['Driver Age', 'Vehicle Age', 'Airbags', 'ESC', 'Brake Assist']
    
    # Normalize values
    case_values = [
        features['age'] / 75 * 100,
        (20 - features['vehicle_age']) / 20 * 100,
        features['airbags'] / 8 * 100,
        100 if features['has_esc'] else 0,
        100 if features['has_brake_assist'] else 0
    ]
    
    avg_values = [
        avg_features.get('age', 45) / 75 * 100,
        (20 - avg_features.get('vehicle_age', 5)) / 20 * 100,
        avg_features.get('airbags', 4) / 8 * 100,
        100 if avg_features.get('has_esc', False) else 0,
        100 if avg_features.get('has_brake_assist', False) else 0
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=case_values,
        theta=categories,
        fill='toself',
        name='This Case',
        line_color='#667eea'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Similar Cases Avg',
        line_color='#ffa500'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        height=250,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# Load system
df, model, idx_claims, idx_no_claims, df_claims, df_no_claims = load_system()
base_rate = df['claim_status'].mean()

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='color: #667eea; margin: 0;'>💬 UnderwriteGPT</h1>
    <p style='color: #6c757d; margin-top: 0.5rem;'>AI-Powered Risk Assessment Assistant</p>
</div>
""", unsafe_allow_html=True)

# Quick start examples
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='color: #6c757d; margin-bottom: 1rem;'>👋 Hi! I'm your underwriting assistant. Try these examples:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 High Risk Case", use_container_width=True, key="ex1"):
            st.session_state.current_query = "22-year-old with 12-year-old Diesel, 2 airbags, no ESC"
    
    with col2:
        if st.button("🟡 Medium Risk Case", use_container_width=True, key="ex2"):
            st.session_state.current_query = "35-year-old with 6-year-old Petrol, 4 airbags, ESC"
    
    with col3:
        if st.button("🟢 Low Risk Case", use_container_width=True, key="ex3"):
            st.session_state.current_query = "45-year-old with 2-year-old Electric, 8 airbags, ESC, brake assist"

# Chat input
query = st.chat_input("Describe the policy you want to assess...")

# Handle query from example buttons
if 'current_query' in st.session_state:
    query = st.session_state.current_query
    del st.session_state.current_query

# Process query
if query:
    st.session_state.query_count += 1
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process
    with st.spinner('🔍 Analyzing...'):
        features = extract_features(query)
        feature_risk = calculate_feature_risk(features, base_rate)
        cases = search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5)
        rag_risk = calculate_weighted_risk(cases, base_rate)
        risk_label, risk_class, combined_risk, emoji = determine_risk_level(rag_risk, feature_risk, base_rate)
        
        claims_found = cases['claim_status'].sum()
        
        # Calculate average features from similar cases
        avg_features = {
            'age': cases.head(10)['customer_age'].mean() if 'customer_age' in cases.columns else 45,
            'vehicle_age': cases.head(10)['vehicle_age'].mean() if 'vehicle_age' in cases.columns else 5,
            'airbags': cases.head(10)['airbags'].mean() if 'airbags' in cases.columns else 4,
            'has_esc': cases.head(10)['is_esc'].mean() > 0.5 if 'is_esc' in cases.columns else False,
            'has_brake_assist': cases.head(10)['is_brake_assist'].mean() > 0.5 if 'is_brake_assist' in cases.columns else False
        }
    
    # Create response
    response = {
        "risk_label": risk_label,
        "risk_class": risk_class,
        "combined_risk": combined_risk,
        "emoji": emoji,
        "features": features,
        "cases": cases,
        "claims_found": claims_found,
        "feature_risk": feature_risk,
        "rag_risk": rag_risk,
        "avg_features": avg_features
    }
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        resp = message["content"]
        
        st.markdown(f"""
        <div class="bot-message">
            <div style='margin-bottom: 1rem;'>
                <strong>UnderwriteGPT:</strong>
            </div>
        """, unsafe_allow_html=True)
        
        # Risk badge and summary
        st.markdown(f"""
            <div style='margin: 1rem 0;'>
                <span class='{resp["risk_class"]}'>{resp["emoji"]} {resp["risk_label"]}</span>
                <span style='margin-left: 1rem; color: #6c757d;'>{resp["combined_risk"]:.1%} risk score</span>
            </div>
            
            <p style='color: #495057; margin: 1rem 0;'>
                Based on analysis of <strong>{resp["claims_found"]}</strong> claims found in <strong>10</strong> similar cases, 
                this policy is assessed as <strong>{resp["risk_label"].lower()}</strong>.
            </p>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{resp["combined_risk"]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            multiplier = resp["combined_risk"] / base_rate if base_rate > 0 else 1.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">vs Base Rate</div>
                <div class="metric-value">{multiplier:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Claims Found</div>
                <div class="metric-value">{resp["claims_found"]}/10</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            confidence = 100 - abs(resp["feature_risk"] - resp["rag_risk"]) * 200
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.plotly_chart(create_risk_gauge(resp["combined_risk"]), use_container_width=True, key=f"gauge_{i}")
        
        with chart_col2:
            st.plotly_chart(create_evidence_chart(resp["cases"]), use_container_width=True, key=f"evidence_{i}")
        
        # Feature comparison
        st.markdown("**📊 Feature Comparison**")
        st.plotly_chart(create_comparison_radar(resp["features"], resp["avg_features"]), 
                       use_container_width=True, key=f"radar_{i}")
        
        # Recommendation
        st.markdown("**💡 Recommendation**")
        
        if resp["combined_risk"] >= 0.12:
            st.markdown("""
            <div class="action-item">
                ⚠️ <strong>Requires manual review</strong> - Premium +30-50%, enhanced documentation, underwriter approval
            </div>
            """, unsafe_allow_html=True)
        elif resp["combined_risk"] >= 0.09:
            st.markdown("""
            <div class="action-item">
                ⚡ <strong>Careful review recommended</strong> - Premium +20-30%, verify vehicle condition
            </div>
            """, unsafe_allow_html=True)
        elif resp["combined_risk"] >= 0.07:
            st.markdown("""
            <div class="action-item">
                ✅ <strong>Standard processing</strong> - Premium +10-20%, regular documentation
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="action-item">
                🌟 <strong>Fast-track eligible</strong> - Competitive pricing, minimal documentation
            </div>
            """, unsafe_allow_html=True)
        
        # Key factors (collapsible)
        with st.expander("🔍 View Key Factors"):
            factors = []
            
            if resp["features"]['age'] < 25:
                factors.append("⚠️ Young driver (+2.5pp risk)")
            elif resp["features"]['age'] > 65:
                factors.append("⚠️ Senior driver (+1.8pp risk)")
            else:
                factors.append("✅ Experienced driver (-0.5pp risk)")
            
            if resp["features"]['vehicle_age'] > 10:
                factors.append("⚠️ Very old vehicle (+3.5pp risk)")
            elif resp["features"]['vehicle_age'] > 7:
                factors.append("⚠️ Aging vehicle (+2.0pp risk)")
            elif resp["features"]['vehicle_age'] <= 3:
                factors.append("✅ New vehicle (-1.5pp risk)")
            
            if resp["features"]['airbags'] >= 6:
                factors.append("✅ Excellent safety (-1.0pp risk)")
            elif resp["features"]['airbags'] <= 2:
                factors.append("⚠️ Basic safety (+1.5pp risk)")
            
            if resp["features"]['has_esc']:
                factors.append("✅ Has ESC (-1.2pp risk)")
            else:
                factors.append("⚠️ No ESC (+1.2pp risk)")
            
            for factor in factors:
                st.markdown(f"• {factor}")
        
        # Similar cases (collapsible)
        with st.expander("📚 View Similar Cases"):
            for j, (idx, row) in enumerate(resp["cases"].head(5).iterrows(), 1):
                status_emoji = "❌" if row['claim_status'] == 1 else "✅"
                status_text = "Claim Filed" if row['claim_status'] == 1 else "No Claim"
                
                summary = row['summary']
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="evidence-header">{j}. {status_emoji} {status_text}</div>
                    <div style="color: #6c757d; font-size: 0.8rem;">{summary}</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
if len(st.session_state.messages) > 0:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div style='text-align: left; color: #888; padding: 1rem 0;'>
            <small>Query {st.session_state.query_count} • {len(df):,} policies analyzed • Response time <200ms</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()