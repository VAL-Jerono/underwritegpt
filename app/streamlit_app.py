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

# Page config
st.set_page_config(
    page_title="UnderwriteGPT - AI Insurance Underwriting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better visual hierarchy
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 75%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: white;
        border: 2px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 90%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .decision-approve { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border-left: 5px solid #28a745;
        margin-bottom: 1.5rem;
    }
    .decision-monitor { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border-left: 5px solid #ffc107;
        margin-bottom: 1.5rem;
    }
    .decision-conditional { 
        background: linear-gradient(135deg, #ffe5cc 0%, #ffd699 100%);
        color: #8b4513; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border-left: 5px solid #ff8c00;
        margin-bottom: 1.5rem;
    }
    .decision-reject { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border-left: 5px solid #dc3545;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .risk-factor-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .evidence-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .evidence-claim {
        border-left: 4px solid #dc3545;
    }
    
    .evidence-no-claim {
        border-left: 4px solid #28a745;
    }
    
    .insight-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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

def extract_features(text: str) -> Dict:
    """Enhanced feature extraction from natural language query"""
    text_lower = text.lower()
    
    # Extract ages
    all_ages = re.findall(r'(\d+)[-\s]?(?:year[-\s]?old|yo|years)', text_lower)
    customer_age = int(all_ages[0]) if all_ages else 35
    vehicle_age = float(all_ages[1]) if len(all_ages) >= 2 else 5.0
    
    # Extract subscription
    sub_match = re.search(r'(\d+)[-\s]?(?:month|mo)', text_lower)
    subscription = int(sub_match.group(1)) if sub_match else 6
    
    # Extract airbags
    airbags_match = re.search(r'(\d+)\s*airbag', text_lower)
    airbags = int(airbags_match.group(1)) if airbags_match else 4
    
    # Extract safety features
    has_esc = 'esc' in text_lower and 'no esc' not in text_lower
    has_brake_assist = 'brake assist' in text_lower or 'brake-assist' in text_lower
    has_tpms = 'tpms' in text_lower
    
    # Extract region context
    is_urban = any(word in text_lower for word in ['urban', 'city', 'metropolitan'])
    
    # Extract fuel type
    fuel_type = 'Petrol'
    if 'diesel' in text_lower:
        fuel_type = 'Diesel'
    elif 'cng' in text_lower:
        fuel_type = 'CNG'
    
    return {
        'customer_age': customer_age,
        'vehicle_age': vehicle_age,
        'subscription_length': subscription,
        'airbags': airbags,
        'has_esc': has_esc,
        'has_brake_assist': has_brake_assist,
        'has_tpms': has_tpms,
        'is_urban': is_urban,
        'fuel_type': fuel_type
    }

def calculate_enhanced_risk_score(features: Dict, base_rate: float = 0.064) -> Dict:
    """
    Calculate risk using validated preprocessing weights from Step 2
    Weights based on actual correlation with claims:
    - subscription: 0.507 (corr: 0.0808)
    - driver: 0.143 (corr: 0.0227)
    - region: 0.139 (corr: 0.0222)
    - vehicle: 0.123 (corr: 0.0195)
    - safety: 0.088 (corr: 0.0141)
    """
    weights = {
        'subscription': 0.507,
        'driver': 0.143,
        'region': 0.139,
        'vehicle': 0.123,
        'safety': 0.088
    }
    
    # Subscription Risk (highest weight)
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
    
    # Driver Risk
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
    
    # Vehicle Risk
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
    
    # Region Risk
    region_risk = 0.55 if features['is_urban'] else 0.45
    region_factor = "Urban region" if features['is_urban'] else "Rural region"
    
    # Safety Risk
    safety_score = (
        features['airbags'] / 6 + 
        (1 if features['has_esc'] else 0) + 
        (1 if features['has_brake_assist'] else 0) +
        (1 if features['has_tpms'] else 0)
    ) / 4
    safety_risk = 1 - safety_score
    
    safety_factors = []
    if features['airbags'] < 4:
        safety_factors.append(f"Limited airbags ({features['airbags']})")
    if not features['has_esc']:
        safety_factors.append("No ESC")
    if not features['has_brake_assist']:
        safety_factors.append("No brake assist")
    
    # Weighted overall risk
    overall_risk = (
        weights['subscription'] * subscription_risk +
        weights['driver'] * driver_risk +
        weights['vehicle'] * vehicle_risk +
        weights['region'] * region_risk +
        weights['safety'] * safety_risk
    )
    
    return {
        'overall_risk': overall_risk,
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
            'region': region_factor,
            'safety': safety_factors if safety_factors else ["Good safety features"]
        }
    }

def search_similar_cases(query: str, model, index, df, k: int = 20) -> pd.DataFrame:
    """Search for similar cases using FAISS"""
    query_vec = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vec, k)
    
    results = df.iloc[indices[0]].copy()
    results['similarity'] = 1 / (1 + distances[0])
    results['distance'] = distances[0]
    
    return results.sort_values('similarity', ascending=False)

def make_enhanced_decision(feature_risk: Dict, similar_cases: pd.DataFrame, base_rate: float = 0.064) -> Dict:
    """
    Enhanced 4-tier decision system aligned with validation thresholds
    From Step 3 validation:
    - Claims avg risk: 0.663
    - No-claims avg risk: 0.582
    - Boundary: ~0.58
    """
    overall_risk = feature_risk['overall_risk']
    claim_rate = similar_cases['claim_status'].mean()
    weighted_claim_rate = (similar_cases['claim_status'] * similar_cases['similarity']).sum() / similar_cases['similarity'].sum()
    
    # Combine feature-based and historical evidence with feature risk floor protection
    # If feature risk is very high (>0.70), don't let historical data override it
    if overall_risk > 0.70:
        final_risk = max((0.60 * overall_risk) + (0.40 * weighted_claim_rate), overall_risk * 0.85)
    else:
        final_risk = (0.40 * overall_risk) + (0.60 * weighted_claim_rate)
    
    # Count claims in top similar cases
    top_5_claims = similar_cases.head(5)['claim_status'].sum()
    top_10_claims = similar_cases.head(10)['claim_status'].sum()
    
    # Decision logic based on validated thresholds with feature risk override
    # CRITICAL: If feature risk alone suggests high risk, don't approve based on sparse historical data
    if final_risk < 0.50 and claim_rate < 0.06 and overall_risk < 0.65:
        return {
            'tier': 'APPROVE',
            'action': '✅ Standard Approval',
            'emoji': '🟢',
            'class': 'decision-approve',
            'premium': '0% (Standard Rates)',
            'details': 'Low risk profile • Competitive rates • Fast-track processing • Minimal documentation',
            'confidence': min(95, 85 + (0.50 - final_risk) * 40),
            'reasoning': f'Risk score ({final_risk:.2%}) well below claims threshold (58.2%). Historical evidence strongly supports approval with {top_5_claims}/5 top matches showing claims.',
            'final_risk': final_risk,
            'feature_risk': overall_risk,
            'historical_risk': weighted_claim_rate
        }
    elif final_risk < 0.58 and claim_rate < 0.09 and overall_risk < 0.70:
        return {
            'tier': 'MONITOR',
            'action': '⚠️ Approve with Monitoring',
            'emoji': '🟡',
            'class': 'decision-monitor',
            'premium': '+15-20% (Moderate Loading)',
            'details': 'Moderate risk • Quarterly policy review • Standard verification • Claim monitoring',
            'confidence': min(85, 70 + (0.58 - final_risk) * 35),
            'reasoning': f'Risk score ({final_risk:.2%}) approaching claims boundary (58.2%). Evidence shows {top_10_claims}/10 similar cases claimed. Warrants careful monitoring.',
            'final_risk': final_risk,
            'feature_risk': overall_risk,
            'historical_risk': weighted_claim_rate
        }
    elif final_risk < 0.75 and claim_rate < 0.15:
        return {
            'tier': 'CONDITIONAL',
            'action': '🔶 Conditional Approval',
            'emoji': '🟠',
            'class': 'decision-conditional',
            'premium': '+30-45% (High Loading)',
            'details': 'High risk • Higher deductible required • Enhanced documentation • Monthly review',
            'confidence': min(75, 60 + (0.75 - final_risk) * 30),
            'reasoning': f'Risk score ({final_risk:.2%}) exceeds no-claims average (58.2%) and approaches claims level (66.3%). Historical data shows {top_10_claims}/10 matches with claims. Requires conditions.',
            'final_risk': final_risk,
            'feature_risk': overall_risk,
            'historical_risk': weighted_claim_rate
        }
    else:
        return {
            'tier': 'REJECT',
            'action': '❌ Decline Application',
            'emoji': '🔴',
            'class': 'decision-reject',
            'premium': 'N/A',
            'details': 'Very high risk • Risk exceeds acceptable thresholds • Consider alternative products',
            'confidence': min(90, 75 + (final_risk - 0.75) * 20),
            'reasoning': f'Risk score ({final_risk:.2%}) significantly exceeds claims average (66.3%). {top_10_claims}/10 similar cases resulted in claims. Risk profile unsuitable for standard coverage.',
            'final_risk': final_risk,
            'feature_risk': overall_risk,
            'historical_risk': weighted_claim_rate
        }

def create_risk_component_chart(risk_analysis: Dict) -> go.Figure:
    """Enhanced risk breakdown showing weighted contributions"""
    components = risk_analysis['components']
    weights = risk_analysis['weights']
    
    # Calculate weighted contributions
    contributions = {k: components[k] * weights[k] for k in components.keys()}
    
    # Sort by contribution
    sorted_items = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#ff8c00', '#ffc107', '#28a745']
    
    for i, (component, contribution) in enumerate(sorted_items):
        fig.add_trace(go.Bar(
            name=component.title(),
            x=[component.title()],
            y=[contribution * 100],
            marker_color=colors[i],
            text=f'{contribution*100:.1f}%',
            textposition='inside',
            textfont=dict(color='white', size=14, family='Arial Black'),
            hovertemplate=f'<b>{component.title()}</b><br>' +
                         f'Raw Score: {components[component]:.2%}<br>' +
                         f'Weight: {weights[component]:.1%}<br>' +
                         f'Contribution: {contribution:.2%}<extra></extra>'
        ))
    
    fig.add_trace(go.Scatter(
        name='Final Risk',
        x=list(sorted_items[0][0].title()),
        y=[risk_analysis['overall_risk'] * 100],
        mode='markers+text',
        marker=dict(size=20, color='#dc3545', symbol='diamond', line=dict(width=2, color='white')),
        text=[f'TOTAL: {risk_analysis["overall_risk"]*100:.1f}%'],
        textposition='top center',
        textfont=dict(size=16, color='#dc3545', family='Arial Black')
    ))
    
    fig.update_layout(
        title=dict(
            text="Risk Score Breakdown (Weighted Components)",
            font=dict(size=16, family='Arial')
        ),
        yaxis_title="Risk Contribution (%)",
        xaxis_title="",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    
    return fig

def create_historical_evidence_chart(similar_cases: pd.DataFrame, decision_tier: str) -> go.Figure:
    """
    Enhanced evidence chart showing WHY the decision was made
    Highlights claim patterns that support the tier
    """
    top_10 = similar_cases.head(10).copy()
    top_10['case_num'] = range(1, len(top_10) + 1)
    top_10['status_label'] = top_10['claim_status'].apply(lambda x: '🔴 CLAIM' if x == 1 else '🟢 NO CLAIM')
    
    # Calculate claim concentration for interpretation
    claims_count = top_10['claim_status'].sum()
    claim_rate = claims_count / len(top_10)
    
    fig = go.Figure()
    
    # Color bars by claim status
    colors = ['#dc3545' if x == 1 else '#28a745' for x in top_10['claim_status']]
    
    fig.add_trace(go.Bar(
        x=top_10['case_num'],
        y=top_10['similarity'],
        marker_color=colors,
        text=top_10['status_label'],
        textposition='outside',
        hovertemplate='<b>Case #%{x}</b><br>' +
                     'Similarity: %{y:.3f}<br>' +
                     'Status: %{text}<br>' +
                     'Risk: ' + top_10['overall_risk_score'].apply(lambda x: f'{x:.2%}') + '<extra></extra>',
        customdata=top_10['overall_risk_score']
    ))
    
    # Add claim rate line
    fig.add_hline(
        y=top_10['similarity'].mean(),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg Similarity: {top_10['similarity'].mean():.3f}",
        annotation_position="right"
    )
    
    # Interpretation based on decision tier
    if decision_tier == 'APPROVE':
        interpretation = f"✅ Only {claims_count}/10 similar cases claimed ({claim_rate:.0%}) - Strong approval signal"
    elif decision_tier == 'MONITOR':
        interpretation = f"⚠️ {claims_count}/10 similar cases claimed ({claim_rate:.0%}) - Moderate risk, monitor closely"
    elif decision_tier == 'CONDITIONAL':
        interpretation = f"🔶 {claims_count}/10 similar cases claimed ({claim_rate:.0%}) - High risk pattern requires conditions"
    else:
        interpretation = f"❌ {claims_count}/10 similar cases claimed ({claim_rate:.0%}) - Unacceptable risk concentration"
    
    fig.update_layout(
        title=dict(
            text=f"Historical Evidence Analysis<br><sub>{interpretation}</sub>",
            font=dict(size=16, family='Arial')
        ),
        xaxis_title="Case Rank (by similarity)",
        yaxis_title="Similarity Score",
        height=350,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    
    return fig

def create_comparison_radar(features: Dict, similar_cases: pd.DataFrame) -> go.Figure:
    """Enhanced radar chart with clear differences"""
    
    # Calculate averages with meaningful scaling
    avg_age_normalized = (similar_cases['customer_age'].mean() - 18) / (70 - 18) * 100
    query_age_normalized = (features['customer_age'] - 18) / (70 - 18) * 100
    
    avg_v_age_normalized = (15 - similar_cases['vehicle_age'].mean()) / 15 * 100
    query_v_age_normalized = (15 - features['vehicle_age']) / 15 * 100
    
    avg_airbags = similar_cases['airbags'].mean() / 6 * 100
    query_airbags = features['airbags'] / 6 * 100
    
    avg_esc = similar_cases['is_esc'].mean() * 100
    query_esc = 100 if features['has_esc'] else 0
    
    avg_sub = similar_cases['subscription_length'].mean() / 12 * 100
    query_sub = features['subscription_length'] / 12 * 100
    
    categories = ['Driver Age<br>(Optimal)', 'Vehicle Age<br>(Newer Better)', 'Airbags', 'ESC', 'Subscription<br>Length']
    
    query_values = [query_age_normalized, query_v_age_normalized, query_airbags, query_esc, query_sub]
    avg_values = [avg_age_normalized, avg_v_age_normalized, avg_airbags, avg_esc, avg_sub]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=query_values,
        theta=categories,
        fill='toself',
        name='Your Application',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.4)',
        hovertemplate='<b>Your Application</b><br>%{theta}: %{r:.1f}/100<extra></extra>'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Similar Cases Average',
        line=dict(color='#ff8c00', width=3),
        fillcolor='rgba(255, 140, 0, 0.25)',
        hovertemplate='<b>Historical Average</b><br>%{theta}: %{r:.1f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0', '25', '50', '75', '100']
            ),
            bgcolor='rgba(240, 240, 240, 0.3)'
        ),
        title=dict(
            text="Profile Comparison (Higher = Better)",
            font=dict(size=16, family='Arial')
        ),
        showlegend=True,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor='white'
    )
    
    return fig

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar with info
with st.sidebar:
    st.markdown("### 🎯 About UnderwriteGPT")
    st.markdown("""
    AI-powered insurance underwriting using:
    - **RAG Architecture**: Retrieval-Augmented Generation
    - **Validated Risk Model**: Based on 58K+ policies
    - **4-Tier Decision System**: Aligned with claim patterns
    
    ---
    
    #### 📊 System Stats
    - **Knowledge Base**: 58,592 policies
    - **Claim Rate**: 6.4%
    - **Risk Threshold**: 58.2% (validated)
    - **Search Speed**: <200ms
    
    ---
    
    #### 🎯 Decision Tiers
    
    **🟢 APPROVE** (Risk < 50%)
    - Standard rates, fast approval
    
    **🟡 MONITOR** (Risk 50-58%)
    - +15-20% premium, regular review
    
    **🟠 CONDITIONAL** (Risk 58-75%)
    - +30-45% premium, conditions required
    
    **🔴 REJECT** (Risk > 75%)
    - Risk exceeds thresholds
    """)

# Load system
with st.spinner('🚀 Loading AI system...'):
    df, model, index, embeddings = load_system()
    base_rate = df['claim_status'].mean()

# Header
st.markdown("""
<div class="main-header">
    <h1 style='margin: 0; font-size: 2.5rem;'>🎯 UnderwriteGPT</h1>
    <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;'>
        AI-Powered Insurance Risk Assessment with Explainable Decisions
    </p>
</div>
""", unsafe_allow_html=True)

# Quick examples with better labels
if len(st.session_state.messages) == 0:
    st.markdown("#### 🚀 Quick Start - Try These Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Low Risk Scenario", use_container_width=True):
            st.session_state.current_query = "42-year-old experienced driver, 3-year-old Petrol sedan, 6 airbags, ESC, brake assist, TPMS, rural area, 12 month subscription"
    
    with col2:
        if st.button("🟡 Medium Risk Scenario", use_container_width=True):
            st.session_state.current_query = "38-year-old driver, 6-year-old Petrol vehicle, 4 airbags, ESC, brake assist, urban area, 6 month subscription"
    
    with col3:
        if st.button("🔴 High Risk Scenario", use_container_width=True):
            st.session_state.current_query = "24-year-old driver, 10-year-old Diesel vehicle, 2 airbags, no ESC, no brake assist, urban area, 3 month subscription"
    
    st.markdown("---")

# Chat input
query = st.chat_input("💬 Describe the insurance application (age, vehicle details, subscription, etc.)")

if 'current_query' in st.session_state:
    query = st.session_state.current_query
    del st.session_state.current_query

# Process query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner('🔍 Analyzing application... Searching 58K+ policies...'):
        # Extract features
        features = extract_features(query)
        
        # Calculate feature-based risk
        risk_analysis = calculate_enhanced_risk_score(features, base_rate)
        
        # Search similar cases
        similar_cases = search_similar_cases(query, model, index, df, k=20)
        
        # Make decision
        decision = make_enhanced_decision(risk_analysis, similar_cases, base_rate)
    
    response = {
        "decision": decision,
        "risk_analysis": risk_analysis,
        "features": features,
        "similar_cases": similar_cases,
        "claims_found": similar_cases.head(10)['claim_status'].sum(),
        "base_rate": base_rate
    }
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display conversation
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>📝 Application Query:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    else:
        resp = message["content"]
        dec = resp["decision"]
        risk_analysis = resp["risk_analysis"]
        features = resp["features"]
        similar_cases = resp["similar_cases"]
        
        # Decision card
        st.markdown(f"""
        <div class="bot-message">
            <div class="{dec['class']}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                            {dec['emoji']} {dec['action']}
                        </div>
                        <div style="font-size: 1rem; margin-bottom: 0.5rem;">
                            <strong>Premium Impact:</strong> {dec['premium']}
                        </div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">
                            {dec['details']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.5rem; font-weight: 700;">
                            {dec['confidence']:.0f}%
                        </div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">
                            Confidence
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reasoning box
        st.markdown(f"""
        <div class="insight-box">
            <strong>💡 Decision Reasoning:</strong><br>
            {dec['reasoning']}
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics row
        st.markdown("#### 📊 Risk Analysis Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            risk_color = "#dc3545" if dec['final_risk'] > 0.66 else "#ffc107" if dec['final_risk'] > 0.58 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Risk</div>
                <div class="metric-value" style="color: {risk_color};">{dec['final_risk']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Feature Risk</div>
                <div class="metric-value">{dec['feature_risk']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Historical Risk</div>
                <div class="metric-value">{dec['historical_risk']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            claims_color = "#dc3545" if resp['claims_found'] > 5 else "#ffc107" if resp['claims_found'] > 2 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Claims Found</div>
                <div class="metric-value" style="color: {claims_color};">{resp['claims_found']}/10</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            multiplier = dec['final_risk'] / resp['base_rate']
            mult_color = "#dc3545" if multiplier > 10 else "#ffc107" if multiplier > 5 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">vs Base Rate</div>
                <div class="metric-value" style="color: {mult_color};">{multiplier:.1f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Key risk factors
        if any(risk_analysis['factors'].values()):
            st.markdown("#### ⚠️ Key Risk Factors Identified")
            
            risk_factors_list = []
            for component, factor in risk_analysis['factors'].items():
                if isinstance(factor, list):
                    for f in factor:
                        if "No " in f or "Limited" in f or "short" in f.lower() or "Young" in f or "Old" in f or "Senior" in f:
                            risk_factors_list.append(f"**{component.title()}**: {f}")
                else:
                    if "short" in factor.lower() or "Young" in factor or "Old" in factor or "Senior" in factor:
                        risk_factors_list.append(f"**{component.title()}**: {factor}")
            
            if risk_factors_list:
                cols = st.columns(min(3, len(risk_factors_list)))
                for idx, factor in enumerate(risk_factors_list[:3]):
                    with cols[idx]:
                        st.markdown(f'<div class="risk-factor-box">⚠️ {factor}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("#### 📈 Visual Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Risk Breakdown", "📊 Historical Evidence", "🔍 Profile Comparison"])
        
        with tab1:
            st.plotly_chart(
                create_risk_component_chart(risk_analysis),
                use_container_width=True,
                key=f"risk_breakdown_{i}"
            )
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>📖 How to Read This Chart:</strong><br>
                • Each bar shows the weighted contribution of that risk factor<br>
                • <strong>Subscription</strong> has the highest weight (50.7%) based on correlation with actual claims<br>
                • The red diamond shows the <strong>final combined risk score</strong><br>
                • Higher bars = greater contribution to overall risk
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.plotly_chart(
                create_historical_evidence_chart(similar_cases, dec['tier']),
                use_container_width=True,
                key=f"evidence_{i}"
            )
            
            # Evidence interpretation
            claims_in_top_5 = similar_cases.head(5)['claim_status'].sum()
            claims_in_top_10 = similar_cases.head(10)['claim_status'].sum()
            
            if dec['tier'] == 'APPROVE':
                evidence_text = f"✅ <strong>Strong Approval Signal:</strong> Only {claims_in_top_10} out of 10 most similar cases resulted in claims. This is below the base claim rate of {resp['base_rate']:.1%}, indicating this profile is safer than average."
            elif dec['tier'] == 'MONITOR':
                evidence_text = f"⚠️ <strong>Moderate Risk Pattern:</strong> {claims_in_top_10} out of 10 similar cases claimed. This is close to the base rate ({resp['base_rate']:.1%}), suggesting careful monitoring is warranted."
            elif dec['tier'] == 'CONDITIONAL':
                evidence_text = f"🔶 <strong>High Risk Pattern:</strong> {claims_in_top_10} out of 10 similar cases resulted in claims. This significantly exceeds the base rate of {resp['base_rate']:.1%}, requiring conditions to offset risk."
            else:
                evidence_text = f"❌ <strong>Unacceptable Risk:</strong> {claims_in_top_10} out of 10 similar cases claimed. This claim concentration ({claims_in_top_10/10:.0%}) is {(claims_in_top_10/10)/resp['base_rate']:.1f}x the base rate, indicating very high risk."
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>📖 Evidence Interpretation:</strong><br>
                {evidence_text}
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.plotly_chart(
                create_comparison_radar(features, similar_cases),
                use_container_width=True,
                key=f"comparison_{i}"
            )
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>📖 Profile Comparison:</strong><br>
                • <strong>Purple area</strong> = Your application profile<br>
                • <strong>Orange area</strong> = Average of similar historical cases<br>
                • Larger area = Better risk profile<br>
                • <strong>Gaps between lines</strong> show where this application differs from similar cases
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed similar cases
        with st.expander("📚 View Detailed Similar Cases (Top 10)", expanded=False):
            st.markdown("""
            <div style="background: #e7f3ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong>💡 Why These Cases Matter:</strong><br>
                These are the 10 most similar historical policies our AI found. Their outcomes help predict 
                the likelihood of a claim for this new application. Pay special attention to cases with 
                high similarity scores (>0.90).
            </div>
            """, unsafe_allow_html=True)
            
            for j, (idx, row) in enumerate(similar_cases.head(10).iterrows(), 1):
                status = row['claim_status']
                similarity = row['similarity']
                risk_score = row['overall_risk_score']
                
                # Determine card class
                card_class = "evidence-claim" if status == 1 else "evidence-no-claim"
                status_emoji = "❌" if status == 1 else "✅"
                status_text = "CLAIM FILED" if status == 1 else "NO CLAIM"
                status_color = "#dc3545" if status == 1 else "#28a745"
                
                # Truncate summary
                summary = row['summary'][:200] + "..." if len(row['summary']) > 200 else row['summary']
                
                st.markdown(f"""
                <div class="evidence-card {card_class}" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                        <div>
                            <strong style="font-size: 1.1rem;">Case #{j}</strong>
                            <span style="margin-left: 1rem; color: {status_color}; font-weight: 700;">
                                {status_emoji} {status_text}
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9rem; color: #6c757d;">Similarity</div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: #667eea;">
                                {similarity:.3f}
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.5rem;">
                        Risk Score: <strong>{risk_score:.2%}</strong> | 
                        Age: {row['customer_age']} | 
                        Vehicle: {row['vehicle_age']:.1f}yr {row['fuel_type']} | 
                        Subscription: {row['subscription_length']:.1f}mo
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.5;">
                        {summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Application details
        with st.expander("📋 Extracted Application Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Driver Information:**")
                st.markdown(f"- Age: {features['customer_age']} years")
                st.markdown(f"- Subscription: {features['subscription_length']} months")
                st.markdown(f"- Region: {'Urban' if features['is_urban'] else 'Rural'}")
                
            with col2:
                st.markdown("**Vehicle Information:**")
                st.markdown(f"- Age: {features['vehicle_age']} years")
                st.markdown(f"- Fuel: {features['fuel_type']}")
                st.markdown(f"- Airbags: {features['airbags']}")
                st.markdown(f"- ESC: {'✅ Yes' if features['has_esc'] else '❌ No'}")
                st.markdown(f"- Brake Assist: {'✅ Yes' if features['has_brake_assist'] else '❌ No'}")
        
        st.markdown("---")

# Footer
if len(st.session_state.messages) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 Start New Assessment", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

# System info footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem; padding: 1rem;'>
    <strong>UnderwriteGPT v2.0</strong> | Powered by RAG + FAISS + Sentence Transformers<br>
    Knowledge Base: 58,592 policies | Base Claim Rate: 6.4% | Risk Model: Validated (Claims avg: 66.3%, No-claims avg: 58.2%)<br>
    <em>This system provides decision support. All approvals should be reviewed by licensed underwriters.</em>
</div>
""", unsafe_allow_html=True)