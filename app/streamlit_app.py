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

# Minimal, clean CSS
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
        # Build indices if missing
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
    """Extract key features from text"""
    age = re.search(r'(\d+)-year-old', text)
    age = int(age.group(1)) if age else 35
    
    v_age = re.search(r'(\d+)-year-old\s+(?:vehicle|car|Petrol|Diesel|Electric)', text)
    v_age = int(v_age.group(1)) if v_age else 5
    
    airbags = re.search(r'(\d+)\s+airbag', text)
    airbags = int(airbags.group(1)) if airbags else 4
    
    has_esc = 'ESC' in text or 'esc' in text.lower()
    
    return {'age': age, 'vehicle_age': v_age, 'airbags': airbags, 'has_esc': has_esc}

def calculate_feature_risk(features, base_rate):
    """Calculate risk from features"""
    risk = base_rate
    
    if features['age'] < 25:
        risk *= 1.3
    elif features['age'] > 60:
        risk *= 1.2
    
    if features['vehicle_age'] > 10:
        risk *= 1.4
    elif features['vehicle_age'] > 5:
        risk *= 1.1
    
    if features['airbags'] < 4:
        risk *= 1.2
    elif features['airbags'] >= 6:
        risk *= 0.9
    
    if not features['has_esc']:
        risk *= 1.1
    
    return min(risk, 1.0)

def search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5):
    """Search both indices"""
    query_vec = model.encode([query])
    
    # Search claims
    dist_c, idx_c = idx_claims.search(query_vec, k)
    results_c = df_claims.iloc[idx_c[0]].copy()
    results_c['distance'] = dist_c[0]
    results_c['source'] = 'claim'
    
    # Search no-claims
    dist_nc, idx_nc = idx_no_claims.search(query_vec, k)
    results_nc = df_no_claims.iloc[idx_nc[0]].copy()
    results_nc['distance'] = dist_nc[0]
    results_nc['source'] = 'no_claim'
    
    # Combine
    combined = pd.concat([results_c, results_nc]).sort_values('distance').reset_index(drop=True)
    
    # Calculate weights
    max_d = combined['distance'].max()
    min_d = combined['distance'].min()
    if max_d > min_d:
        combined['weight'] = 1 - ((combined['distance'] - min_d) / (max_d - min_d))
    else:
        combined['weight'] = 1.0
    
    return combined

def calculate_weighted_risk(cases):
    """Calculate weighted risk"""
    weighted_sum = (cases['claim_status'] * cases['weight']).sum()
    total_weight = cases['weight'].sum()
    return weighted_sum / total_weight if total_weight > 0 else 0

def determine_risk_level(rag_risk, feature_risk, base_rate):
    """Determine final risk level"""
    combined = (0.6 * rag_risk) + (0.4 * feature_risk)
    multiplier = combined / base_rate
    
    if multiplier >= 2.5:
        return "🔴 HIGH RISK", "risk-high", combined, multiplier
    elif multiplier >= 2.0:
        return "🟠 MEDIUM-HIGH RISK", "risk-medium-high", combined, multiplier
    elif multiplier >= 1.5:
        return "🟡 MEDIUM RISK", "risk-medium", combined, multiplier
    else:
        return "🟢 LOW RISK", "risk-low", combined, multiplier

# Load system
df, model, idx_claims, idx_no_claims, df_claims, df_no_claims = load_system()
base_rate = df['claim_status'].mean()

# Header
st.markdown('<p class="main-header">🎯 UnderwriteGPT</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Fast • Smart • Explainable Risk Assessment</p>', unsafe_allow_html=True)

# Quick examples in columns
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 High Risk Example", use_container_width=True):
        st.session_state.query = "22-year-old with 12-year-old Diesel, 2 airbags, no ESC"
with col2:
    if st.button("🟡 Medium Risk Example", use_container_width=True):
        st.session_state.query = "35-year-old with 6-year-old Petrol, 4 airbags, ESC"
with col3:
    if st.button("🟢 Low Risk Example", use_container_width=True):
        st.session_state.query = "45-year-old with 2-year-old Electric, 8 airbags, ESC"

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
        feature_risk = calculate_feature_risk(features, base_rate)
        
        # Search
        cases = search_balanced(query, model, idx_claims, idx_no_claims, df_claims, df_no_claims, k=5)
        
        # Calculate risk
        rag_risk = calculate_weighted_risk(cases)
        risk_label, risk_class, combined_risk, multiplier = determine_risk_level(rag_risk, feature_risk, base_rate)
        
        claims_found = cases['claim_status'].sum()
    
    st.markdown("---")
    
    # Risk badge
    st.markdown(f'<div class="{risk_class}">{risk_label}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Combined Risk", f"{combined_risk:.1%}")
    col2.metric("Risk Multiplier", f"{multiplier:.1f}x")
    col3.metric("Claims Found", f"{claims_found}/10")
    col4.metric("Base Rate", f"{base_rate:.1%}")
    
    st.markdown("---")
    
    # Two columns: Actions and Evidence
    left, right = st.columns([1, 1])
    
    with left:
        st.markdown("### 📋 Recommended Actions")
        
        if multiplier >= 2.5:
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
        
        elif multiplier >= 2.0:
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
        
        elif multiplier >= 1.5:
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
            factors.append("⚠️ Young driver (higher risk)")
        elif features['age'] > 60:
            factors.append("⚠️ Senior driver")
        else:
            factors.append("✅ Experienced driver")
        
        if features['vehicle_age'] > 10:
            factors.append("⚠️ Old vehicle (12+ years)")
        elif features['vehicle_age'] > 5:
            factors.append("⚠️ Aging vehicle (6-10 years)")
        else:
            factors.append("✅ Newer vehicle")
        
        if features['airbags'] >= 6:
            factors.append("✅ Excellent safety (6+ airbags)")
        elif features['airbags'] >= 4:
            factors.append("✅ Good safety (4+ airbags)")
        else:
            factors.append("⚠️ Basic safety (2-3 airbags)")
        
        if features['has_esc']:
            factors.append("✅ Has ESC")
        else:
            factors.append("⚠️ No ESC")
        
        for f in factors:
            st.markdown(f"• {f}")
    
    with right:
        st.markdown("### 📊 Evidence from Similar Cases")
        
        st.markdown(f"""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
<b>Sample:</b> 5 claims + 5 no-claims<br>
<b>Weighted Risk:</b> {rag_risk:.1%}<br>
<b>Your Risk:</b> {combined_risk:.1%} ({multiplier:.1f}x base)
</div>
""", unsafe_allow_html=True)
        
        # Show top 6 cases
        for i, (idx, row) in enumerate(cases.head(6).iterrows(), 1):
            status = "❌ Claim" if row['claim_status'] == 1 else "✅ No Claim"
            similarity = row['weight']
            
            # Truncate summary
            summary = row['summary']
            if len(summary) > 150:
                summary = summary[:150] + "..."
            
            st.markdown(f"""
<div class="case-summary">
<b>{i}. {status}</b> | Match: {similarity:.2f}<br>
<small>{summary}</small>
</div>
""", unsafe_allow_html=True)
        
        # Similarity chart
        fig = go.Figure()
        colors = cases.head(10)['claim_status'].map({0: '#44cc44', 1: '#ff4444'})
        fig.add_trace(go.Bar(
            x=list(range(1, 11)),
            y=cases.head(10)['weight'],
            marker_color=colors,
            text=cases.head(10)['claim_status'].map({0: '✅', 1: '❌'}),
            textposition='auto'
        ))
        fig.update_layout(
            title="Similarity Weights (Sorted)",
            xaxis_title="Case",
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
Feature Risk: {feature_risk:.2%}
RAG Risk (Weighted): {rag_risk:.2%}

SIMILAR CASES: {claims_found}/10 had claims

TOP CASES:
"""
        for i, (idx, row) in enumerate(cases.head(10).iterrows(), 1):
            status = "CLAIM" if row['claim_status'] == 1 else "NO CLAIM"
            report += f"\n{i}. [{status}] Weight: {row['weight']:.3f}\n   {row['summary']}\n"
        
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
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
<small>UnderwriteGPT • Balanced RAG System • {total:,} policies analyzed</small>
</div>
""".format(total=len(df)), unsafe_allow_html=True)