import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
from pathlib import Path
import re

# Page config
st.set_page_config(
    page_title="UnderwriteGPT - Smart Risk Assessment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .risk-badge {
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b, #d62728);
        color: white;
    }
    
    .risk-medium-high { 
        background: linear-gradient(135deg, #ff9f43, #ff7f0e);
        color: white;
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #ffd93d, #ffc107);
        color: #333;
    }
    
    .risk-medium-low { 
        background: linear-gradient(135deg, #95e1d3, #38ada9);
        color: white;
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #6bcf7f, #2ca02c);
        color: white;
    }
    
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    
    .tip-box {
        background: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ff7f0e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    """Load all required resources with balanced indices"""
    try:
        df = pd.read_csv('data/processed/data_with_summaries.csv')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to load dual indices
        try:
            index_claims = faiss.read_index('models/faiss_index_claims.bin')
            index_no_claims = faiss.read_index('models/faiss_index_no_claims.bin')
            
            # Load split dataframes
            claim_mask = df['claim_status'] == 1
            df_claims = df[claim_mask].copy()
            df_no_claims = df[~claim_mask].copy()
            
            return df, model, index_claims, index_no_claims, df_claims, df_no_claims, True
        except:
            st.warning("⚠️ Balanced indices not found. Creating them now...")
            
            # Create balanced indices
            embeddings = np.load('models/embeddings.npy')
            claim_mask = df['claim_status'] == 1
            
            df_claims = df[claim_mask].copy()
            df_no_claims = df[~claim_mask].copy()
            
            embeddings_claims = embeddings[claim_mask]
            embeddings_no_claims = embeddings[~claim_mask]
            
            dimension = embeddings.shape[1]
            
            index_claims = faiss.IndexFlatL2(dimension)
            index_claims.add(embeddings_claims)
            
            index_no_claims = faiss.IndexFlatL2(dimension)
            index_no_claims.add(embeddings_no_claims)
            
            # Save for future use
            faiss.write_index(index_claims, 'models/faiss_index_claims.bin')
            faiss.write_index(index_no_claims, 'models/faiss_index_no_claims.bin')
            
            st.success("✅ Balanced indices created and saved!")
            
            return df, model, index_claims, index_no_claims, df_claims, df_no_claims, True
            
    except Exception as e:
        st.error(f"⚠️ Error loading resources: {e}")
        st.info("📝 Please run the data preparation notebooks first.")
        st.stop()

def extract_features_from_text(text):
    """Extract key features from natural language query"""
    features = {
        'age': None,
        'vehicle_age': None,
        'airbags': None,
        'fuel_type': None,
        'has_esc': False,
        'has_brake_assist': False
    }
    
    age_match = re.search(r'(\d+)-year-old', text)
    if age_match:
        features['age'] = int(age_match.group(1))
    
    vehicle_age_match = re.search(r'(\d+)-year-old\s+(?:vehicle|car|Petrol|Diesel|Electric)', text)
    if vehicle_age_match:
        features['vehicle_age'] = int(vehicle_age_match.group(1))
    
    airbag_match = re.search(r'(\d+)\s+airbag', text)
    if airbag_match:
        features['airbags'] = int(airbag_match.group(1))
    
    if 'Electric' in text or 'electric' in text:
        features['fuel_type'] = 'Electric'
    elif 'Diesel' in text or 'diesel' in text:
        features['fuel_type'] = 'Diesel'
    elif 'Petrol' in text or 'petrol' in text:
        features['fuel_type'] = 'Petrol'
    
    features['has_esc'] = 'ESC' in text or 'esc' in text.lower()
    features['has_brake_assist'] = 'brake assist' in text.lower()
    
    return features

def calculate_feature_risk(features, df):
    """Calculate risk based on extracted features"""
    risk_factors = []
    base_rate = df['claim_status'].mean()
    estimated_risk = base_rate
    
    if features['age']:
        if features['age'] < 25:
            risk_factors.append(f"👤 Young driver (age {features['age']}) - Higher risk group")
            estimated_risk *= 1.3
        elif features['age'] > 60:
            risk_factors.append(f"👴 Senior driver (age {features['age']}) - Elevated risk")
            estimated_risk *= 1.2
        else:
            risk_factors.append(f"👤 Experienced driver (age {features['age']}) - Standard risk")
    
    if features['vehicle_age']:
        if features['vehicle_age'] > 10:
            risk_factors.append(f"🚗 Old vehicle ({features['vehicle_age']} years) - Higher maintenance issues")
            estimated_risk *= 1.4
        elif features['vehicle_age'] > 5:
            risk_factors.append(f"🚗 Aging vehicle ({features['vehicle_age']} years) - Moderate risk")
            estimated_risk *= 1.1
        else:
            risk_factors.append(f"🚗 Newer vehicle ({features['vehicle_age']} years) - Lower risk")
    
    if features['airbags']:
        if features['airbags'] >= 6:
            risk_factors.append(f"✅ Excellent safety ({features['airbags']} airbags) - Risk reducer")
            estimated_risk *= 0.9
        elif features['airbags'] >= 4:
            risk_factors.append(f"✅ Good safety ({features['airbags']} airbags) - Standard")
        else:
            risk_factors.append(f"⚠️ Basic safety ({features['airbags']} airbags) - Risk factor")
            estimated_risk *= 1.2
    
    if features['has_esc']:
        risk_factors.append("✅ Has ESC (Electronic Stability Control) - Good safety")
        estimated_risk *= 0.95
    else:
        risk_factors.append("❌ No ESC - Missing key safety feature")
    
    if features['has_brake_assist']:
        risk_factors.append("✅ Has Brake Assist - Enhanced safety")
        estimated_risk *= 0.95
    
    if features['fuel_type'] == 'Electric':
        risk_factors.append("⚡ Electric vehicle - Modern, safer")
        estimated_risk *= 0.9
    
    return {
        'estimated_risk': min(estimated_risk, 1.0),
        'factors': risk_factors,
        'base_rate': base_rate
    }

def balanced_search(query_text, model, index_claims, index_no_claims, df_claims, df_no_claims, k_per_group=5):
    """Balanced search using dual indices"""
    query_vector = model.encode([query_text])
    
    # Search claims index
    distances_claims, indices_claims = index_claims.search(query_vector, k_per_group)
    results_claims = df_claims.iloc[indices_claims[0]].copy()
    results_claims['similarity_distance'] = distances_claims[0]
    results_claims['source'] = 'claims'
    
    # Search no-claims index
    distances_no_claims, indices_no_claims = index_no_claims.search(query_vector, k_per_group)
    results_no_claims = df_no_claims.iloc[indices_no_claims[0]].copy()
    results_no_claims['similarity_distance'] = distances_no_claims[0]
    results_no_claims['source'] = 'no_claims'
    
    # Combine and sort by similarity
    combined = pd.concat([results_claims, results_no_claims], ignore_index=True)
    combined = combined.sort_values('similarity_distance').reset_index(drop=True)
    combined['similarity_score'] = 1 / (1 + combined['similarity_distance'])
    
    return combined

def calculate_weighted_risk(similar_cases):
    """Calculate weighted risk score"""
    max_dist = similar_cases['similarity_distance'].max()
    min_dist = similar_cases['similarity_distance'].min()
    
    if max_dist > min_dist:
        normalized_dist = (similar_cases['similarity_distance'] - min_dist) / (max_dist - min_dist)
    else:
        normalized_dist = pd.Series([0.5] * len(similar_cases))
    
    similarity_weights = 1 - normalized_dist
    
    weighted_sum = (similar_cases['claim_status'] * similarity_weights).sum()
    total_weight = similarity_weights.sum()
    weighted_rate = weighted_sum / total_weight if total_weight > 0 else 0
    
    return {
        'weighted_rate': weighted_rate,
        'regular_rate': similar_cases['claim_status'].mean(),
        'total_cases': len(similar_cases),
        'total_claims': similar_cases['claim_status'].sum(),
        'weights': similarity_weights
    }

def determine_risk_level(weighted_rate, feature_risk, base_rate):
    """Determine risk level with adjusted thresholds"""
    # Combine: 40% features, 60% RAG
    combined_risk = (0.4 * feature_risk) + (0.6 * weighted_rate)
    
    # Thresholds based on base rate multipliers
    if combined_risk >= base_rate * 2.5:  # 2.5x base rate
        return "HIGH RISK", "risk-high", "🔴", combined_risk
    elif combined_risk >= base_rate * 2.0:  # 2x base rate
        return "MEDIUM-HIGH RISK", "risk-medium-high", "🟠", combined_risk
    elif combined_risk >= base_rate * 1.5:  # 1.5x base rate
        return "MEDIUM RISK", "risk-medium", "🟡", combined_risk
    elif combined_risk >= base_rate * 1.2:  # 1.2x base rate
        return "MEDIUM-LOW RISK", "risk-medium-low", "🟢", combined_risk
    else:
        return "LOW RISK", "risk-low", "🟢", combined_risk

# Load resources
df, model, index_claims, index_no_claims, df_claims, df_no_claims, has_dual_index = load_resources()

# Header
st.markdown('<p class="main-header">🎯 UnderwriteGPT</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Insurance Risk Assessment | Balanced • Smart • Explainable</p>', unsafe_allow_html=True)

# Show balance info
if has_dual_index:
    st.markdown(f"""
    <div class='tip-box'>
    ✅ <b>Balanced Assessment Active</b><br>
    Using dual-index approach: {len(df_claims):,} claims + {len(df_no_claims):,} no-claims<br>
    Base claim rate: {df['claim_status'].mean():.2%} | Every search returns 50/50 split for fair comparison
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=80)
    st.markdown("### ⚙️ Settings")
    
    k_per_group = st.slider(
        "Cases per group (claims/no-claims)",
        min_value=3,
        max_value=10,
        value=5,
        help="Returns this many from claims AND no-claims for balanced comparison"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Policies", f"{len(df):,}")
        st.metric("Claims", f"{len(df_claims):,}")
    with col2:
        st.metric("Base Rate", f"{df['claim_status'].mean():.2%}")
        st.metric("No Claims", f"{len(df_no_claims):,}")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Examples")
    
    if st.button("🔴 High Risk Example", use_container_width=True):
        st.session_state.query = "22-year-old with 12-year-old Diesel vehicle, 2 airbags, no ESC, no brake assist"
    
    if st.button("🟡 Medium Risk Example", use_container_width=True):
        st.session_state.query = "35-year-old with 6-year-old Petrol Honda Civic, 4 airbags, ESC"
    
    if st.button("🟢 Low Risk Example", use_container_width=True):
        st.session_state.query = "45-year-old with 2-year-old Electric Tesla, 8 airbags, ESC, brake assist, parking sensors"
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.85rem; color: #666;'>
    💡 <b>Balanced Assessment:</b><br>
    • Searches claims & no-claims separately<br>
    • Forces 50/50 representation<br>
    • Eliminates 6.4% base rate bias<br>
    • More accurate risk differentiation
    </div>
    """, unsafe_allow_html=True)

# Main input area
st.markdown("### 📝 Describe the Insurance Case")

query_text = st.text_area(
    label="Enter case details:",
    value=st.session_state.get('query', ''),
    height=120,
    placeholder="Example: 28-year-old with 6-year-old Petrol Honda Civic, 4 airbags, ESC, brake assist",
    help="Include: driver age, vehicle age, fuel type, safety features"
)

col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col2:
    analyze_btn = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
with col3:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.query = ''
    st.rerun()

if analyze_btn and query_text:
    with st.spinner('🧠 Analyzing with balanced assessment...'):
        # Extract features
        features = extract_features_from_text(query_text)
        feature_analysis = calculate_feature_risk(features, df)
        
        # Balanced RAG search
        similar_cases = balanced_search(
            query_text, model, index_claims, index_no_claims, 
            df_claims, df_no_claims, k_per_group
        )
        
        # Calculate weighted risk
        risk_metrics = calculate_weighted_risk(similar_cases)
        
        # Determine risk level
        risk_level, risk_class, emoji, combined_risk = determine_risk_level(
            risk_metrics['weighted_rate'],
            feature_analysis['estimated_risk'],
            feature_analysis['base_rate']
        )
    
    st.markdown("---")
    
    # Risk Badge
    st.markdown(f"""
    <div class='risk-badge {risk_class}'>
        {emoji} {risk_level}
    </div>
    """, unsafe_allow_html=True)
    
    # Risk multiplier warning
    multiplier = combined_risk / feature_analysis['base_rate']
    if multiplier >= 2.5:
        st.markdown(f"""
        <div class='warning-box'>
        ⚠️ <b>Alert:</b> This profile is <b>{multiplier:.1f}x</b> more likely to claim than average ({feature_analysis['base_rate']:.1%})
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🎯 Combined Risk", f"{combined_risk:.2%}")
    with col2:
        st.metric("📊 Feature Risk", f"{feature_analysis['estimated_risk']:.2%}")
    with col3:
        st.metric("🔍 RAG Risk", f"{risk_metrics['weighted_rate']:.2%}")
    with col4:
        st.metric("📁 Claims Found", f"{risk_metrics['total_claims']}/{risk_metrics['total_cases']}")
    with col5:
        st.metric("📈 Risk Multiplier", f"{multiplier:.2f}x")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧩 Risk Factor Analysis")
        st.markdown(f"<div class='tip-box'><b>Base claim rate:</b> {feature_analysis['base_rate']:.2%}<br><b>This case:</b> {combined_risk:.2%} ({multiplier:.1f}x base)</div>", unsafe_allow_html=True)
        
        for factor in feature_analysis['factors']:
            st.markdown(f"• {factor}")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=combined_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 20}},
            delta={'reference': feature_analysis['base_rate'] * 100, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 30], 'ticksuffix': '%'},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, feature_analysis['base_rate'] * 120], 'color': "#c8e6c9"},
                    {'range': [feature_analysis['base_rate'] * 120, feature_analysis['base_rate'] * 150], 'color': "#fff9c4"},
                    {'range': [feature_analysis['base_rate'] * 150, feature_analysis['base_rate'] * 200], 'color': "#ffcc80"},
                    {'range': [feature_analysis['base_rate'] * 200, 30], 'color': "#ef9a9a"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': combined_risk * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Balanced Sample Analysis")
        
        st.markdown(f"""
        <div class='tip-box'>
        ✅ <b>Balanced Retrieval:</b><br>
        • {k_per_group} from claims index<br>
        • {k_per_group} from no-claims index<br>
        • Weighted by similarity<br>
        • Regular rate: {risk_metrics['regular_rate']:.2%}<br>
        • Weighted rate: {risk_metrics['weighted_rate']:.2%}
        </div>
        """, unsafe_allow_html=True)
        
        # Claim distribution pie chart
        claims = risk_metrics['total_claims']
        no_claims = risk_metrics['total_cases'] - claims
        
        fig = go.Figure(data=[go.Pie(
            labels=['✅ No Claim', '❌ Claim Filed'],
            values=[no_claims, claims],
            marker_colors=['#2ca02c', '#d62728'],
            hole=0.5,
            textinfo='label+value',
            textfont_size=14
        )])
        fig.update_layout(
            title=f"Outcomes in Retrieved Cases",
            height=250,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Similarity distribution
        fig = go.Figure()
        colors = similar_cases['claim_status'].map({0: '#2ca02c', 1: '#d62728'})
        fig.add_trace(go.Bar(
            x=list(range(1, len(similar_cases)+1)),
            y=similar_cases['similarity_score'],
            marker_color=colors,
            text=similar_cases['claim_status'].map({0: '✅', 1: '❌'}),
            textposition='auto',
            hovertemplate='<b>Case %{x}</b><br>Similarity: %{y:.3f}<br>Source: %{customdata}<extra></extra>',
            customdata=similar_cases['source']
        ))
        fig.update_layout(
            title="Similarity Scores (Sorted)",
            xaxis_title="Case Number",
            yaxis_title="Similarity Score",
            height=250,
            margin=dict(l=20, r=20, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cases
    st.markdown("---")
    st.markdown(f"### 📋 Top {min(10, len(similar_cases))} Most Similar Cases (Balanced Sample)")
    
    for i, (idx, row) in enumerate(similar_cases.head(10).iterrows(), 1):
        status_icon = "❌ CLAIM FILED" if row['claim_status'] == 1 else "✅ NO CLAIM"
        source_badge = "🔴 Claims Index" if row['source'] == 'claims' else "🟢 No-Claims Index"
        
        with st.expander(f"**Case {i}:** {status_icon} | Similarity: {row['similarity_score']:.3f} | {source_badge}", expanded=(i<=2)):
            st.markdown(f"**📄 Summary:** {row['summary']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Policy ID", row['policy_id'])
            with col2:
                st.metric("Customer Age", f"{row['customer_age']} years")
            with col3:
                st.metric("Vehicle Age", f"{row['vehicle_age']} years")
            with col4:
                st.metric("Airbags", int(row['airbags']))
    
    # Recommendation
    st.markdown("---")
    st.markdown("### 💡 Underwriting Recommendation")
    
    if combined_risk >= feature_analysis['base_rate'] * 2.5:
        st.error(f"""
        #### 🔴 HIGH RISK - MANUAL REVIEW REQUIRED
        
        **Risk Score:** {combined_risk:.2%} (Base rate: {feature_analysis['base_rate']:.2%})  
        **Risk Multiplier:** {multiplier:.1f}x more likely to claim than average
        
        **Evidence:**
        - Similar cases show {risk_metrics['weighted_rate']:.2%} weighted claim rate
        - Feature analysis suggests {feature_analysis['estimated_risk']:.2%} risk
        - Found {risk_metrics['total_claims']}/{risk_metrics['total_cases']} claims in balanced sample
        
        **Recommended Actions:**
        - ⚠️ **REQUIRE manual underwriter review**
        - 💰 Premium adjustment: **+30% to +50%**
        - 📄 Request additional documentation
        - 🔍 Verify all safety features thoroughly
        - 📋 Consider stricter policy terms or coverage limitations
        - 💳 Higher deductible recommended
        """)
    elif combined_risk >= feature_analysis['base_rate'] * 2.0:
        st.warning(f"""
        #### 🟠 MEDIUM-HIGH RISK - CAREFUL REVIEW
        
        **Risk Score:** {combined_risk:.2%} (Base rate: {feature_analysis['base_rate']:.2%})  
        **Risk Multiplier:** {multiplier:.1f}x more likely to claim than average
        
        **Evidence:**
        - Similar cases show {risk_metrics['weighted_rate']:.2%} weighted claim rate
        - Feature analysis suggests {feature_analysis['estimated_risk']:.2%} risk
        - Found {risk_metrics['total_claims']}/{risk_metrics['total_cases']} claims in balanced sample
        
        **Recommended Actions:**
        - 👀 Manual review strongly recommended
        - 💰 Premium adjustment: **+20% to +30%**
        - ✅ Verify safety features and vehicle condition
        - 📋 Standard terms with enhanced monitoring
        - 📄 Request proof of safety features
        """)
    elif combined_risk >= feature_analysis['base_rate'] * 1.5:
        st.info(f"""
        #### 🟡 MEDIUM RISK - STANDARD PROCESSING WITH VERIFICATION
        
        **Risk Score:** {combined_risk:.2%} (Base rate: {feature_analysis['base_rate']:.2%})  
        **Risk Multiplier:** {multiplier:.1f}x more likely to claim than average
        
        **Evidence:**
        - Similar cases show {risk_metrics['weighted_rate']:.2%} weighted claim rate
        - Feature analysis suggests {feature_analysis['estimated_risk']:.2%} risk
        - Found {risk_metrics['total_claims']}/{risk_metrics['total_cases']} claims in balanced sample
        
        **Recommended Actions:**
        - ✅ Standard processing acceptable
        - 💰 Premium adjustment: **+10% to +20%**
        - 🔍 Verify key risk factors
        - 📋 Regular policy terms apply
        - 📄 Standard documentation required
        """)
    elif combined_risk >= feature_analysis['base_rate'] * 1.2:
        st.success(f"""
        #### 🟢 MEDIUM-LOW RISK - STANDARD PROCESSING
        
        **Risk Score:** {combined_risk:.2%} (Base rate: {feature_analysis['base_rate']:.2%})  
        **Risk Multiplier:** {multiplier:.1f}x (near base rate)
        
        **Evidence:**
        - Similar cases show {risk_metrics['weighted_rate']:.2%} weighted claim rate
        - Feature analysis suggests {feature_analysis['estimated_risk']:.2%} risk
        - Found {risk_metrics['total_claims']}/{risk_metrics['total_cases']} claims in balanced sample
        
        **Recommended Actions:**
        - ✅ Standard processing
        - 💰 Standard premium rates apply
        - 📄 Standard documentation
        - 📋 Regular policy terms
        """)
    else:
        st.success(f"""
        #### 🟢 LOW RISK - FAST TRACK ELIGIBLE
        
        **Risk Score:** {combined_risk:.2%} (Base rate: {feature_analysis['base_rate']:.2%})  
        **Risk Multiplier:** {multiplier:.1f}x (below base rate)
        
        **Evidence:**
        - Similar cases show {risk_metrics['weighted_rate']:.2%} weighted claim rate
        - Feature analysis suggests {feature_analysis['estimated_risk']:.2%} risk
        - Found {risk_metrics['total_claims']}/{risk_metrics['total_cases']} claims in balanced sample
        
        **Recommended Actions:**
        - ✅ **Fast-track approval eligible**
        - 💰 Standard or **competitive premium rates**
        - 📄 Minimal documentation required
        - 🎁 Consider loyalty discount eligibility
        - 📋 Preferred customer treatment
        """)
    
    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download CSV Data", use_container_width=True):
            csv = similar_cases.to_csv(index=False)
            st.download_button(
                label="Download Similar Cases CSV",
                data=csv,
                file_name=f"similar_cases_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        report = f"""
UNDERWRITEGPT BALANCED RISK ASSESSMENT REPORT
{'='*70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

QUERY: {query_text}

RISK ASSESSMENT
{'='*70}
Risk Level: {risk_level}
Combined Risk Score: {combined_risk:.2%}
Risk Multiplier: {multiplier:.2f}x base rate

Component Scores:
- Feature-Based Risk: {feature_analysis['estimated_risk']:.2%}
- RAG-Based Risk (Weighted): {risk_metrics['weighted_rate']:.2%}
- RAG-Based Risk (Regular): {risk_metrics['regular_rate']:.2%}
- Base Dataset Rate: {feature_analysis['base_rate']:.2%}

RISK FACTORS IDENTIFIED
{'='*70}
"""
        for factor in feature_analysis['factors']:
            report += f"{factor}\n"
        
        report += f"""
BALANCED SAMPLE ANALYSIS
{'='*70}
Total Cases Analyzed: {risk_metrics['total_cases']} ({k_per_group} claims + {k_per_group} no-claims)
Claims Found: {risk_metrics['total_claims']}
Weighted Claim Rate: {risk_metrics['weighted_rate']:.2%}
Regular Claim Rate: {risk_metrics['regular_rate']:.2%}

SIMILAR CASES (TOP 10):
{'='*70}
"""
        for i, (idx, row) in enumerate(similar_cases.head(10).iterrows(), 1):
            status = "CLAIM" if row['claim_status'] == 1 else "NO CLAIM"
            report += f"\n{i}. [{status}] Similarity: {row['similarity_score']:.3f} | Source: {row['source']}\n"
            report += f"   {row['summary']}\n"
        
        st.download_button(
            label="📄 Download Full Report",
            data=report,
            file_name=f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

elif analyze_btn:
    st.warning("⚠️ Please enter a case description to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p style='font-size: 0.9rem;'>Built with ❤️ using Streamlit • Powered by sentence-transformers & FAISS</p>
    <p style='font-size: 0.85rem;'>UnderwriteGPT - Balanced RAG for Fair Risk Assessment</p>
</div>
""", unsafe_allow_html=True)