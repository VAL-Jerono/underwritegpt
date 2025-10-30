import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
import re

# Page config
st.set_page_config(
    page_title="UnderwriteGPT",
    page_icon="🎯",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .risk-high { 
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-medium-high { 
        background: #ff9933;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-medium { 
        background: #ffcc00;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-low { 
        background: #44cc44;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
    }
    
    .action-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .case-summary {
        background: white;
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load all resources"""
    df = pd.read_csv('data/processed/data_with_summaries.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        index_claims = faiss.read_index('models/faiss_claims_index.bin')
        index_no_claims = faiss.read_index('models/faiss_no_claims_index.bin')
    except:
        embeddings = np.load('models/embeddings.npy')
        claim_mask = df['claim_status'] == 1
        
        embeddings_claims = embeddings[claim_mask]
        embeddings_no_claims = embeddings[~claim_mask]
        
        dimension = embeddings.shape[1]
        index_claims = faiss.IndexFlatL2(dimension)
        index_claims.add(embeddings_claims)
        
        index_no_claims = faiss.IndexFlatL2(dimension)
        index_no_claims.add(embeddings_no_claims)
        
        faiss.write_index(index_claims, 'models/faiss_claims_index.bin')
        faiss.write_index(index_no_claims, 'models/faiss_no_claims_index.bin')
    
    claim_mask = df['claim_status'] == 1
    df_claims = df[claim_mask].reset_index(drop=True)
    df_no_claims = df[~claim_mask].reset_index(drop=True)
    
    return df, model, index_claims, index_no_claims, df_claims, df_no_claims

def extract_features(text):
    """Extract key features - COMPLETELY REWRITTEN"""
    text_lower = text.lower()
    
    # Find ALL numbers followed by "year-old" or "year old"
    all_ages = re.findall(r'(\d+)[-\s]?year[-\s]?old', text_lower)
    
    # First number is driver age, second (if exists) is vehicle age
    if len(all_ages) >= 2:
        age = int(all_ages[0])
        v_age = float(all_ages[1])
    elif len(all_ages) == 1:
        age = int(all_ages[0])
        v_age = 5.0  # Default
    else:
        age = 35
        v_age = 5.0
    
    # Airbags
    airbags_match = re.search(r'(\d+)\s*airbag', text_lower)
    airbags = int(airbags_match.group(1)) if airbags_match else 4
    
    # Safety features - exact word boundary matching
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
    """Calculate risk from features - REALISTIC adjustments"""
    risk_adjustment = 0.0
    
    # Age risk
    if features['age'] < 25:
        risk_adjustment += 0.025  # +2.5pp
    elif features['age'] < 30:
        risk_adjustment += 0.012  # +1.2pp
    elif features['age'] > 65:
        risk_adjustment += 0.018  # +1.8pp
    elif features['age'] > 60:
        risk_adjustment += 0.008  # +0.8pp
    else:
        risk_adjustment -= 0.005  # -0.5pp
    
    # Vehicle age risk (MOST IMPORTANT FACTOR)
    if features['vehicle_age'] > 10:
        risk_adjustment += 0.035  # +3.5pp
    elif features['vehicle_age'] > 7:
        risk_adjustment += 0.020  # +2.0pp
    elif features['vehicle_age'] > 5:
        risk_adjustment += 0.010  # +1.0pp
    elif features['vehicle_age'] <= 3:
        risk_adjustment -= 0.015  # -1.5pp
    
    # Airbags
    if features['airbags'] <= 2:
        risk_adjustment += 0.015  # +1.5pp
    elif features['airbags'] >= 6:
        risk_adjustment -= 0.010  # -1.0pp
    
    # Safety features
    if not features['has_esc']:
        risk_adjustment += 0.012  # +1.2pp
    if not features['has_brake_assist']:
        risk_adjustment += 0.008  # +0.8pp
    
    feature_risk = base_rate + risk_adjustment
    feature_risk = max(0.01, min(0.30, feature_risk))
    
    return feature_risk

def search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5):
    """Search both indices with proper weighting"""
    query_vec = model.encode([query])
    
    # Search both indices
    dist_c, idx_c = idx_claims.search(query_vec, k)
    results_c = df_claims.iloc[idx_c[0]].copy()
    results_c['distance'] = dist_c[0]
    results_c['source'] = 'claim'
    
    dist_nc, idx_nc = idx_no_claims.search(query_vec, k)
    results_nc = df_no_claims.iloc[idx_nc[0]].copy()
    results_nc['distance'] = dist_nc[0]
    results_nc['source'] = 'no_claim'
    
    # Combine
    combined = pd.concat([results_c, results_nc]).reset_index(drop=True)
    
    # Simple inverse distance weighting
    combined['weight'] = 1.0 / (1.0 + combined['distance'])
    
    # Sort by distance
    combined = combined.sort_values('distance').reset_index(drop=True)
    
    return combined

def calculate_weighted_risk(cases, base_rate):
    """Calculate weighted risk with better calibration"""
    # Weighted average
    weighted_sum = (cases['claim_status'] * cases['weight']).sum()
    total_weight = cases['weight'].sum()
    
    if total_weight == 0:
        return base_rate
    
    weighted_risk = weighted_sum / total_weight
    
    # Adjust towards base rate to prevent extreme values from forced 50/50 sampling
    # The forced sampling means we expect ~50% base proportion in results
    # We need to scale this back to the true base rate
    
    # If weighted_risk is close to 0.5, that's "average" similarity
    # Map 0.5 -> base_rate, <0.5 -> lower, >0.5 -> higher
    
    deviation = weighted_risk - 0.5
    adjusted_risk = base_rate + (deviation * base_rate * 2)
    
    # Clamp to reasonable range
    adjusted_risk = max(0.01, min(0.25, adjusted_risk))
    
    return adjusted_risk

def determine_risk_level(rag_risk, feature_risk, base_rate):
    """Determine final risk level - Feature-heavy weighting"""
    # Give more weight to features since RAG has forced sampling bias
    combined = (0.70 * feature_risk) + (0.30 * rag_risk)
    
    if combined >= 0.12:  # 12%+ (2x base rate)
        return "🔴 HIGH RISK", "risk-high", combined
    elif combined >= 0.09:  # 9-12% (1.4-2x base rate)
        return "🟠 MEDIUM-HIGH RISK", "risk-medium-high", combined
    elif combined >= 0.07:  # 7-9% (1.1-1.4x base rate)
        return "🟡 MEDIUM RISK", "risk-medium", combined
    else:  # <7%
        return "🟢 LOW RISK", "risk-low", combined

# Load system
df, model, idx_claims, idx_no_claims, df_claims, df_no_claims = load_system()
base_rate = df['claim_status'].mean()

# Header
st.markdown('<p class="main-header">🎯 UnderwriteGPT</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Fast • Smart • Explainable Risk Assessment</p>', unsafe_allow_html=True)

# Quick examples
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 High Risk Example", use_container_width=True):
        st.session_state.query = "22-year-old with 12-year-old Diesel, 2 airbags, no ESC"
with col2:
    if st.button("🟡 Medium Risk Example", use_container_width=True):
        st.session_state.query = "35-year-old with 6-year-old Petrol, 4 airbags, ESC"
with col3:
    if st.button("🟢 Low Risk Example", use_container_width=True):
        st.session_state.query = "45-year-old with 2-year-old Electric, 8 airbags, ESC, brake assist"

# Main input
query = st.text_input(
    "Describe the case:",
    value=st.session_state.get('query', ''),
    placeholder="e.g., 28-year-old with 6-year-old Petrol Honda, 4 airbags, ESC",
    help="Include: driver age, vehicle age, fuel type, safety features"
)

analyze = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)

if analyze and query:
    with st.spinner('Analyzing...'):
        # Extract features
        features = extract_features(query)
        
        # Calculate risks
        feature_risk = calculate_feature_risk(features, base_rate)
        
        # Search
        cases = search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5)
        
        # Calculate RAG risk
        rag_risk = calculate_weighted_risk(cases, base_rate)
        
        # Determine level
        risk_label, risk_class, combined_risk = determine_risk_level(rag_risk, feature_risk, base_rate)
        
        claims_found = cases['claim_status'].sum()
        multiplier = combined_risk / base_rate if base_rate > 0 else 1.0
    
    st.markdown("---")
    
    # Risk badge
    st.markdown(f'<div class="{risk_class}">{risk_label}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Debug info
    with st.expander("🔧 Debug Info - Feature Extraction Check"):
        st.write("**Query:**", query)
        st.write("**Extracted Features:**", features)
        st.write("---")
        st.write(f"**Base Rate:** {base_rate:.2%}")
        st.write(f"**Feature Risk:** {feature_risk:.2%} ({(feature_risk/base_rate):.2f}x base) - 70% weight")
        st.write(f"**RAG Risk:** {rag_risk:.2%} ({(rag_risk/base_rate):.2f}x base) - 30% weight")
        st.write(f"**Combined Risk:** {combined_risk:.2%}")
        st.write(f"**Final Multiplier:** {multiplier:.2f}x base rate")
        st.write("---")
        st.write(f"**Risk Thresholds:**")
        st.write(f"- HIGH: ≥12.0%")
        st.write(f"- MED-HIGH: ≥9.0%")
        st.write(f"- MEDIUM: ≥7.0%")
        st.write(f"- LOW: <7.0%")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Combined Risk", f"{combined_risk:.1%}")
    col2.metric("Risk Multiplier", f"{multiplier:.1f}x")
    col3.metric("Claims Found", f"{claims_found}/10")
    col4.metric("Base Rate", f"{base_rate:.1%}")
    
    st.markdown("---")
    
    # Two columns
    left, right = st.columns([1, 1])
    
    with left:
        st.markdown("### 📋 Recommended Actions")
        
        if combined_risk >= 0.12:
            st.markdown("""
<div class="action-card">
<b>⚠️ MANUAL REVIEW REQUIRED</b><br><br>
<b>Premium:</b> +30% to +50%<br>
<b>Actions:</b><br>
• Require underwriter approval<br>
• Request additional documentation<br>
• Verify all safety features<br>
• Consider higher deductible
</div>
""", unsafe_allow_html=True)
        
        elif combined_risk >= 0.09:
            st.markdown("""
<div class="action-card">
<b>⚠️ CAREFUL REVIEW</b><br><br>
<b>Premium:</b> +20% to +30%<br>
<b>Actions:</b><br>
• Manual review recommended<br>
• Verify vehicle condition<br>
• Enhanced documentation<br>
• Standard monitoring
</div>
""", unsafe_allow_html=True)
        
        elif combined_risk >= 0.07:
            st.markdown("""
<div class="action-card">
<b>📋 STANDARD PROCESSING</b><br><br>
<b>Premium:</b> +10% to +20%<br>
<b>Actions:</b><br>
• Standard verification<br>
• Regular documentation<br>
• Normal processing time
</div>
""", unsafe_allow_html=True)
        
        else:
            st.markdown("""
<div class="action-card">
<b>✅ FAST TRACK ELIGIBLE</b><br><br>
<b>Premium:</b> Standard or competitive<br>
<b>Actions:</b><br>
• Fast-track approval<br>
• Minimal documentation<br>
• Consider loyalty discount<br>
• Preferred customer
</div>
""", unsafe_allow_html=True)
        
        # Key factors
        st.markdown("### 🔍 Key Factors")
        
        factors = []
        if features['age'] < 25:
            factors.append(f"⚠️ Young driver ({features['age']} years, +2.5pp)")
        elif features['age'] < 30:
            factors.append(f"⚠️ Young driver ({features['age']} years, +1.2pp)")
        elif features['age'] > 65:
            factors.append(f"⚠️ Senior driver ({features['age']} years, +1.8pp)")
        elif features['age'] > 60:
            factors.append(f"⚠️ Senior driver ({features['age']} years, +0.8pp)")
        else:
            factors.append(f"✅ Experienced driver ({features['age']} years, -0.5pp)")
        
        if features['vehicle_age'] > 10:
            factors.append(f"⚠️ OLD vehicle ({features['vehicle_age']:.1f} years, +3.5pp)")
        elif features['vehicle_age'] > 7:
            factors.append(f"⚠️ Aging vehicle ({features['vehicle_age']:.1f} years, +2.0pp)")
        elif features['vehicle_age'] > 5:
            factors.append(f"⚡ Medium age vehicle ({features['vehicle_age']:.1f} years, +1.0pp)")
        elif features['vehicle_age'] <= 3:
            factors.append(f"✅ New vehicle ({features['vehicle_age']:.1f} years, -1.5pp)")
        else:
            factors.append(f"✅ Newer vehicle ({features['vehicle_age']:.1f} years, neutral)")
        
        if features['airbags'] >= 6:
            factors.append(f"✅ Excellent safety ({features['airbags']} airbags, -1.0pp)")
        elif features['airbags'] >= 4:
            factors.append(f"✅ Good safety ({features['airbags']} airbags, neutral)")
        else:
            factors.append(f"⚠️ Basic safety ({features['airbags']} airbags, +1.5pp)")
        
        if features['has_esc']:
            factors.append("✅ Has ESC (-1.2pp)")
        else:
            factors.append("⚠️ No ESC (+1.2pp)")
            
        if features['has_brake_assist']:
            factors.append("✅ Has brake assist (-0.8pp)")
        else:
            factors.append("⚠️ No brake assist (+0.8pp)")
        
        st.markdown("*pp = percentage points adjustment*")
        for f in factors:
            st.markdown(f"• {f}")
    
    with right:
        st.markdown("### 📊 Evidence from Similar Cases")
        
        st.markdown(f"""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
<b>Sample:</b> 5 claims + 5 no-claims<br>
<b>Feature Risk:</b> {feature_risk:.1%} (70% weight)<br>
<b>RAG Risk:</b> {rag_risk:.1%} (30% weight)<br>
<b>Combined:</b> {combined_risk:.1%} ({multiplier:.1f}x base)
</div>
""", unsafe_allow_html=True)
        
        # Show top cases
        for i, (idx, row) in enumerate(cases.head(8).iterrows(), 1):
            status = "❌ Claim" if row['claim_status'] == 1 else "✅ No Claim"
            weight = row['weight']
            distance = row['distance']
            
            summary = row['summary']
            if len(summary) > 140:
                summary = summary[:140] + "..."
            
            st.markdown(f"""
<div class="case-summary">
<b>{i}. {status}</b> | Weight: {weight:.3f} | Distance: {distance:.2f}<br>
<small>{summary}</small>
</div>
""", unsafe_allow_html=True)
        
        # Weight distribution chart
        fig = go.Figure()
        colors = cases.head(10)['claim_status'].map({0: '#44cc44', 1: '#ff4444'})
        fig.add_trace(go.Bar(
            x=list(range(1, 11)),
            y=cases.head(10)['weight'],
            marker_color=colors,
            text=cases.head(10)['claim_status'].map({0: '✅', 1: '❌'}),
            textposition='auto',
            hovertemplate='<b>Case %{x}</b><br>Weight: %{y:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title="Case Weights Distribution",
            xaxis_title="Case Rank",
            yaxis_title="Weight",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = cases.to_csv(index=False)
        st.download_button(
            "📥 Download Cases (CSV)",
            csv,
            f"cases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        report = f"""UNDERWRITEGPT RISK ASSESSMENT
{'='*60}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

QUERY: {query}

RISK ASSESSMENT:
Risk Level: {risk_label}
Combined Risk: {combined_risk:.2%}
Risk Multiplier: {multiplier:.1f}x base rate
Base Rate: {base_rate:.2%}

COMPONENTS:
Feature Risk: {feature_risk:.2%} (70% weight)
RAG Risk: {rag_risk:.2%} (30% weight)
Combined: {combined_risk:.2%}

EXTRACTED FEATURES:
Driver Age: {features['age']}
Vehicle Age: {features['vehicle_age']} years
Airbags: {features['airbags']}
ESC: {'Yes' if features['has_esc'] else 'No'}
Brake Assist: {'Yes' if features['has_brake_assist'] else 'No'}

SIMILAR CASES: {claims_found}/10 had claims

TOP CASES:
"""
        for i, (idx, row) in enumerate(cases.head(10).iterrows(), 1):
            status = "CLAIM" if row['claim_status'] == 1 else "NO CLAIM"
            report += f"\n{i}. [{status}] Weight: {row['weight']:.3f} Distance: {row['distance']:.2f}\n   {row['summary']}\n"
        
        st.download_button(
            "📄 Download Report (TXT)",
            report,
            f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            use_container_width=True
        )

elif analyze:
    st.warning("⚠️ Please enter a case description")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #888; padding: 1rem;'>
<small>UnderwriteGPT • Calibrated Risk System • {len(df):,} policies analyzed</small>
</div>
""", unsafe_allow_html=True)