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
    page_title="UnderwriteGPT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal, clean CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 80%;
    }
    
    .bot-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 85%;
    }
    
    .decision-approve { background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; }
    .decision-monitor { background: #fff3cd; color: #856404; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; }
    .decision-conditional { background: #ffe5cc; color: #8b4513; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff8c00; }
    .decision-reject { background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; }
    
    .metric-compact {
        text-align: center;
        padding: 0.5rem;
    }
    .metric-compact .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-compact .label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load all resources with LlamaIndex ready"""
    df = pd.read_csv('data/processed/train_data_with_summaries.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        index = faiss.read_index('models/faiss_index.bin')
        embeddings = np.load('models/embeddings.npy')
        
        if len(embeddings) != len(df):
            st.error(f"❌ Dimension mismatch: {len(embeddings)} embeddings vs {len(df)} rows")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error loading index: {e}")
        st.info("💡 Run notebook 06_ragframeworkindex.ipynb first")
        st.stop()
    
    return df, model, index, embeddings

def extract_features(text):
    """Extract features from query"""
    text_lower = text.lower()
    
    all_ages = re.findall(r'(\d+)[-\s]?year[-\s]?old', text_lower)
    age = int(all_ages[0]) if all_ages else 35
    v_age = float(all_ages[1]) if len(all_ages) >= 2 else 5.0
    
    airbags_match = re.search(r'(\d+)\s*airbag', text_lower)
    airbags = int(airbags_match.group(1)) if airbags_match else 4
    
    has_esc = 'esc' in text_lower and 'no esc' not in text_lower
    has_brake_assist = 'brake assist' in text_lower
    
    return {
        'age': age, 
        'vehicle_age': v_age, 
        'airbags': airbags, 
        'has_esc': has_esc,
        'has_brake_assist': has_brake_assist
    }

def search_cases(query, model, index, df, k=10):
    """Search for similar cases"""
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    
    results = df.iloc[indices[0]].copy()
    results['similarity'] = 1 / (1 + distances[0])  # Convert distance to similarity
    
    return results

def calculate_risk_score(features, similar_cases, base_rate=0.064):
    """Calculate final risk score with clear logic"""
    
    # 1. Feature-based risk (40% weight)
    feature_risk = base_rate
    
    # Age impact
    if features['age'] < 25:
        feature_risk += 0.030
    elif features['age'] < 30:
        feature_risk += 0.015
    elif features['age'] > 65:
        feature_risk += 0.020
    
    # Vehicle age impact
    if features['vehicle_age'] > 10:
        feature_risk += 0.040
    elif features['vehicle_age'] > 7:
        feature_risk += 0.025
    elif features['vehicle_age'] <= 3:
        feature_risk -= 0.010
    
    # Safety impact
    if features['airbags'] <= 2:
        feature_risk += 0.020
    elif features['airbags'] >= 6:
        feature_risk -= 0.015
    
    if not features['has_esc']:
        feature_risk += 0.015
    if not features['has_brake_assist']:
        feature_risk += 0.010
    
    # 2. Historical evidence (60% weight)
    claim_rate = similar_cases['claim_status'].mean()
    weighted_claim_rate = (similar_cases['claim_status'] * similar_cases['similarity']).sum() / similar_cases['similarity'].sum()
    
    # 3. Combine
    final_risk = (0.40 * feature_risk) + (0.60 * weighted_claim_rate)
    
    return final_risk, feature_risk, weighted_claim_rate

def make_decision(risk_score, claim_rate, confidence):
    """Clear 4-tier decision logic"""
    
    if risk_score < 0.06:
        return {
            'tier': 'APPROVE',
            'action': '✅ Standard Approval',
            'details': 'Competitive rates • Fast processing • Standard documentation',
            'class': 'decision-approve',
            'emoji': '🟢'
        }
    elif risk_score < 0.09:
        return {
            'tier': 'MONITOR',
            'action': '⚠️ Approve with Monitoring',
            'details': '+10-15% premium • Regular policy review • Standard verification',
            'class': 'decision-monitor',
            'emoji': '🟡'
        }
    elif risk_score < 0.12:
        return {
            'tier': 'CONDITIONAL',
            'action': '🔶 Conditional Approval',
            'details': '+25-40% premium • Higher deductible • Enhanced documentation required',
            'class': 'decision-conditional',
            'emoji': '🟠'
        }
    else:
        return {
            'tier': 'REJECT',
            'action': '❌ Decline Application',
            'details': 'Risk exceeds acceptable thresholds • Refer to alternative coverage options',
            'class': 'decision-reject',
            'emoji': '🔴'
        }

def create_comparison_chart(features, similar_cases):
    """Visual comparison: Query vs Similar Cases"""
    
    # Calculate averages from similar cases
    avg_age = similar_cases['customer_age'].mean()
    avg_v_age = similar_cases['vehicle_age'].mean()
    avg_airbags = similar_cases['airbags'].mean()
    avg_esc = similar_cases['is_esc'].mean() * 100
    
    categories = ['Driver Age', 'Vehicle Age', 'Airbags', 'ESC']
    
    # Normalize to 0-100 scale for comparison
    query_values = [
        (features['age'] / 75) * 100,
        ((20 - features['vehicle_age']) / 20) * 100,  # Inverse (newer = better)
        (features['airbags'] / 8) * 100,
        100 if features['has_esc'] else 0
    ]
    
    avg_values = [
        (avg_age / 75) * 100,
        ((20 - avg_v_age) / 20) * 100,
        (avg_airbags / 8) * 100,
        avg_esc
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=query_values,
        theta=categories,
        fill='toself',
        name='Your Query',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Similar Cases Avg',
        line=dict(color='#ff8c00', width=2),
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )
        ),
        showlegend=True,
        height=280,
        margin=dict(l=60, r=60, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig

def create_risk_breakdown_chart(feature_risk, historical_risk, final_risk):
    """Show how risk components combine"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Feature Analysis',
        x=['Risk Components'],
        y=[feature_risk * 100],
        marker_color='#667eea',
        text=[f'{feature_risk*100:.1f}%'],
        textposition='inside',
        width=0.3
    ))
    
    fig.add_trace(go.Bar(
        name='Historical Evidence',
        x=['Risk Components'],
        y=[historical_risk * 100],
        marker_color='#ff8c00',
        text=[f'{historical_risk*100:.1f}%'],
        textposition='inside',
        width=0.3
    ))
    
    fig.add_trace(go.Scatter(
        name='Final Risk Score',
        x=['Risk Components'],
        y=[final_risk * 100],
        mode='markers+text',
        marker=dict(size=15, color='#dc3545', symbol='diamond'),
        text=[f'{final_risk*100:.1f}%'],
        textposition='top center',
        textfont=dict(size=14, color='#dc3545', family='Arial Black')
    ))
    
    fig.update_layout(
        title="Risk Score Breakdown",
        yaxis_title="Risk (%)",
        height=250,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        barmode='group'
    )
    
    return fig

def create_evidence_distribution(similar_cases):
    """Show claim distribution with similarity weights"""
    
    top_cases = similar_cases.head(10).copy()
    top_cases['status'] = top_cases['claim_status'].apply(lambda x: 'Claimed' if x == 1 else 'No Claim')
    top_cases['case_num'] = range(1, len(top_cases) + 1)
    
    fig = go.Figure()
    
    # Color by status
    colors = ['#dc3545' if x == 1 else '#28a745' for x in top_cases['claim_status']]
    
    fig.add_trace(go.Bar(
        x=top_cases['case_num'],
        y=top_cases['similarity'],
        marker_color=colors,
        text=[f"{'🔴' if x == 1 else '🟢'}" for x in top_cases['claim_status']],
        textposition='outside',
        hovertemplate='<b>Case %{x}</b><br>Similarity: %{y:.2f}<br>Status: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 10 Similar Cases by Relevance",
        xaxis_title="Case Rank",
        yaxis_title="Similarity Score",
        height=250,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    
    return fig

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load system
df, model, index, embeddings = load_system()
base_rate = df['claim_status'].mean()

# Header
st.markdown("""
<div style='text-align: center; padding: 1.5rem 0 1rem 0;'>
    <h1 style='color: #667eea; margin: 0;'>🎯 UnderwriteGPT</h1>
    <p style='color: #6c757d; margin-top: 0.5rem;'>AI-Powered Insurance Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Quick examples
if len(st.session_state.messages) == 0:
    st.markdown("<div style='text-align: center; margin: 1rem 0;'><p style='color: #6c757d;'>Try these examples:</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Low Risk", use_container_width=True):
            st.session_state.current_query = "45-year-old, 2-year-old Petrol, 6 airbags, ESC, brake assist"
    
    with col2:
        if st.button("🟡 Medium Risk", use_container_width=True):
            st.session_state.current_query = "35-year-old, 7-year-old Diesel, 4 airbags, ESC"
    
    with col3:
        if st.button("🔴 High Risk", use_container_width=True):
            st.session_state.current_query = "23-year-old, 12-year-old vehicle, 2 airbags, no ESC"

# Chat input
query = st.chat_input("Describe the policy application...")

if 'current_query' in st.session_state:
    query = st.session_state.current_query
    del st.session_state.current_query

# Process query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner('🔍 Analyzing...'):
        features = extract_features(query)
        similar_cases = search_cases(query, model, index, df, k=10)
        risk_score, feature_risk, historical_risk = calculate_risk_score(features, similar_cases, base_rate)
        
        claims_found = similar_cases['claim_status'].sum()
        confidence = 100 - (abs(feature_risk - historical_risk) * 300)
        confidence = max(50, min(95, confidence))
        
        decision = make_decision(risk_score, historical_risk, confidence)
    
    response = {
        "decision": decision,
        "risk_score": risk_score,
        "feature_risk": feature_risk,
        "historical_risk": historical_risk,
        "confidence": confidence,
        "features": features,
        "similar_cases": similar_cases,
        "claims_found": claims_found
    }
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    else:
        resp = message["content"]
        dec = resp["decision"]
        
        st.markdown(f"""
        <div class="bot-message">
            <div class="{dec['class']}" style="margin-bottom: 1rem;">
                <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {dec['emoji']} {dec['action']}
                </div>
                <div style="font-size: 0.9rem;">{dec['details']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Compact metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-compact">
                <div class="value">{resp["risk_score"]:.1%}</div>
                <div class="label">Final Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-compact">
                <div class="value">{resp["claims_found"]}/10</div>
                <div class="label">Claims Found</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            multiplier = resp["risk_score"] / base_rate
            st.markdown(f"""
            <div class="metric-compact">
                <div class="value">{multiplier:.1f}x</div>
                <div class="label">vs Base</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-compact">
                <div class="value">{resp["confidence"]:.0f}%</div>
                <div class="label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.plotly_chart(
                create_comparison_chart(resp["features"], resp["similar_cases"]), 
                use_container_width=True, 
                key=f"comp_{i}"
            )
        
        with chart_col2:
            st.plotly_chart(
                create_risk_breakdown_chart(resp["feature_risk"], resp["historical_risk"], resp["risk_score"]),
                use_container_width=True,
                key=f"breakdown_{i}"
            )
        
        # Evidence distribution
        st.plotly_chart(
            create_evidence_distribution(resp["similar_cases"]),
            use_container_width=True,
            key=f"evidence_{i}"
        )
        
        # Expandable details
        with st.expander("📋 View Similar Cases"):
            for j, (idx, row) in enumerate(resp["similar_cases"].head(5).iterrows(), 1):
                status = "❌ Claim" if row['claim_status'] == 1 else "✅ No Claim"
                sim = row['similarity']
                summary = row['summary'][:120] + "..." if len(row['summary']) > 120 else row['summary']
                
                st.markdown(f"""
                **{j}. {status}** (Similarity: {sim:.2f})  
                {summary}
                """)

# Footer
if len(st.session_state.messages) > 0:
    st.markdown("---")
    if st.button("🔄 New Assessment", use_container_width=True):
        st.session_state.messages = []
        st.rerun()