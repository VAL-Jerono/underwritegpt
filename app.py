"""
UNDERWRITING ASSISTANT - STREAMLIT WEB APPLICATION
===================================================
Interactive LLM-powered assistant for insurance agents

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer
import faiss
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Underwriting Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .approval-approved {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .approval-declined {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .risk-low {color: #28a745; font-weight: bold;}
    .risk-moderate {color: #ffc107; font-weight: bold;}
    .risk-high {color: #fd7e14; font-weight: bold;}
    .risk-very-high {color: #dc3545; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODELS (Cached for performance)
# ============================================================================

@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    with open('models/underwriting_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_names']

@st.cache_resource
def load_rag_system():
    """Load the RAG system with FAISS index"""
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load FAISS index
    index = faiss.read_index('models/faiss_index/customer.index')
    
    # Load sentences and metadata
    with open('models/faiss_index/sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)
    
    with open('models/faiss_index/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return embedding_model, index, sentences, metadata

@st.cache_data
def load_historical_data():
    """Load historical customer data"""
    return pd.read_csv('data/cleaned_insurance_data.csv')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_score(data):
    """Calculate risk score from customer data"""
    score = (
        data['speeding_violations'] * 2.0 +
        data['duis'] * 10.0 +
        data['past_accidents'] * 3.0 +
        (5 - data['age_numeric']) * 1.5 +
        (5 - data['experience_numeric']) * 1.0 +
        (1 if data['annual_mileage'] > 13000 else 0) * 2.0 +
        (1 if data['credit_score'] < 0.4 else 0) * 2.0 +
        (0 if data['vehicle_year'] == 'after 2015' else 1) * 1.0
    )
    return score

def categorize_risk(score, duis, total_violations):
    """Categorize risk level"""
    if duis > 0 or total_violations >= 3:
        return "Very High"
    elif score <= 10.5:
        return "Low"
    elif score <= 13.0:
        return "Moderate"
    elif score <= 20.5:
        return "High"
    else:
        return "Very High"

def search_similar_customers(query, embedding_model, index, sentences, metadata, top_k=3):
    """Search for similar customers"""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            'profile': sentences[idx],
            'metadata': metadata[idx],
            'similarity': float(1 / (1 + dist))
        })
    return results

def get_recommendation(claim_prob, risk_category):
    """Generate underwriting recommendation"""
    if claim_prob < 0.3 and risk_category in ['Low', 'Moderate']:
        return "✅ **RECOMMEND APPROVAL** - Standard premium tier"
    elif claim_prob < 0.5 and risk_category in ['Moderate', 'High']:
        return "⚠️ **CONDITIONAL APPROVAL** - Elevated premium tier recommended"
    else:
        return "❌ **RECOMMEND DECLINE** - High risk profile, manual review required"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("<h1 class='main-header'>🛡️ Underwriting Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Insurance Risk Assessment</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models..."):
        ml_model, feature_names = load_ml_model()
        embedding_model, faiss_index, sentences, metadata = load_rag_system()
        historical_data = load_historical_data()
    
    # Sidebar - Input Method Selection
    st.sidebar.header("📋 Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Entry", "Upload Customer Data", "Quick Search"]
    )
    
    # ========================================================================
    # METHOD 1: MANUAL ENTRY
    # ========================================================================
    if input_method == "Manual Entry":
        st.header("Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Demographics")
            age = st.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'])
            gender = st.selectbox("Gender", ['male', 'female'])
            race = st.selectbox("Race", ['majority', 'minority'])
            
        with col2:
            st.subheader("🎓 Background")
            education = st.selectbox("Education", ['none', 'high school', 'university'])
            income = st.selectbox("Income Level", ['poverty', 'working class', 'middle class', 'upper class'])
            experience = st.selectbox("Driving Experience", ['0-9y', '10-19y', '20-29y', '30y+'])
            
        with col3:
            st.subheader("💳 Financial")
            credit_score = st.slider("Credit Score", 0.0, 1.0, 0.5, 0.01)
            vehicle_ownership = st.selectbox("Vehicle Ownership", ['Owns Vehicle', 'Does Not Own'])
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("🚗 Vehicle Details")
            vehicle_type = st.selectbox("Vehicle Type", ['sedan', 'sports car'])
            vehicle_year = st.selectbox("Vehicle Year", ['after 2015', 'before 2015'])
            annual_mileage = st.number_input("Annual Mileage", 2000, 25000, 12000, 1000)
            
        with col5:
            st.subheader("⚠️ Driving Record")
            speeding_violations = st.number_input("Speeding Violations", 0, 20, 0)
            duis = st.number_input("DUIs", 0, 5, 0)
            past_accidents = st.number_input("Past Accidents", 0, 15, 0)
        
        st.subheader("👨‍👩‍👧‍👦 Personal Status")
        col6, col7 = st.columns(2)
        with col6:
            married = st.selectbox("Marital Status", ['Single', 'Married'])
        with col7:
            children = st.number_input("Number of Children", 0, 5, 0)
        
        # Analyze button
        if st.button("🔍 Analyze Customer", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer profile..."):
                # Prepare data
                age_map = {'16-25': 1, '26-39': 2, '40-64': 3, '65+': 4}
                exp_map = {'0-9y': 1, '10-19y': 2, '20-29y': 3, '30y+': 4}
                edu_map = {'none': 0, 'high school': 1, 'university': 2}
                income_map = {'poverty': 0, 'working class': 1, 'middle class': 2, 'upper class': 3}
                
                customer_data = {
                    'AGE_NUMERIC': age_map[age],
                    'GENDER_ENCODED': 1 if gender == 'male' else 0,
                    'RACE_ENCODED': 0 if race == 'majority' else 1,
                    'EXPERIENCE_NUMERIC': exp_map[experience],
                    'EDUCATION_NUMERIC': edu_map[education],
                    'INCOME_NUMERIC': income_map[income],
                    'CREDIT_SCORE': credit_score,
                    'VEHICLE_OWNERSHIP': 1 if vehicle_ownership == 'Owns Vehicle' else 0,
                    'VEHICLE_NEW': 1 if vehicle_year == 'after 2015' else 0,
                    'VEHICLE_TYPE_ENCODED': 1 if vehicle_type == 'sports car' else 0,
                    'ANNUAL_MILEAGE': annual_mileage,
                    'MARRIED': 1 if married == 'Married' else 0,
                    'CHILDREN': children,
                    'SPEEDING_VIOLATIONS': speeding_violations,
                    'DUIS': duis,
                    'PAST_ACCIDENTS': past_accidents,
                    'TOTAL_VIOLATIONS': speeding_violations + duis + past_accidents,
                    'HAS_VIOLATIONS': 1 if (speeding_violations + duis + past_accidents) > 0 else 0,
                    'HAS_DUI': 1 if duis > 0 else 0,
                    'HIGH_MILEAGE': 1 if annual_mileage > 13000 else 0,
                    'YOUNG_INEXPERIENCED': 1 if (age == '16-25' and experience == '0-9y') else 0,
                    'LOW_CREDIT': 1 if credit_score < 0.4 else 0,
                }
                
                # Calculate risk score
                risk_score = calculate_risk_score({
                    'speeding_violations': speeding_violations,
                    'duis': duis,
                    'past_accidents': past_accidents,
                    'age_numeric': age_map[age],
                    'experience_numeric': exp_map[experience],
                    'annual_mileage': annual_mileage,
                    'credit_score': credit_score,
                    'vehicle_year': vehicle_year
                })
                
                customer_data['RISK_SCORE'] = risk_score
                
                # Categorize risk
                risk_category = categorize_risk(
                    risk_score, 
                    duis, 
                    customer_data['TOTAL_VIOLATIONS']
                )
                customer_data['RISK_CATEGORY_ENCODED'] = {'High': 0, 'Low': 1, 'Moderate': 2, 'Very High': 3}[risk_category]
                
                # ML Prediction
                X = pd.DataFrame([customer_data])[feature_names]
                prediction = ml_model.predict(X)[0]
                probability = ml_model.predict_proba(X)[0]
                claim_prob = probability[1]
                
                # Determine approval
                approval_status = "APPROVED" if claim_prob < 0.5 else "DECLINED"
                
                # Display results
                display_results(
                    approval_status, claim_prob, risk_category, risk_score,
                    customer_data, embedding_model, faiss_index, sentences, metadata
                )
    
    # ========================================================================
    # METHOD 2: UPLOAD CSV
    # ========================================================================
    elif input_method == "Upload Customer Data":
        st.header("Upload Customer Data")
        st.info("Upload a CSV file with customer information for batch processing")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            st.dataframe(df.head())
            
            if st.button("Process All Customers", type="primary"):
                st.info("Batch processing feature - Coming soon!")
    
    # ========================================================================
    # METHOD 3: QUICK SEARCH
    # ========================================================================
    else:  # Quick Search
        st.header("🔍 Search Similar Customer Profiles")
        st.write("Search our database using natural language queries")
        
        query = st.text_area(
            "Enter your search query:",
            "Young driver with 2 speeding violations and low credit score",
            height=100
        )
        
        if st.button("Search", type="primary"):
            with st.spinner("Searching database..."):
                results = search_similar_customers(
                    query, embedding_model, faiss_index, sentences, metadata, top_k=5
                )
                
                st.subheader("Top 5 Similar Customers")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Customer {result['metadata']['customer_id']} - {result['metadata']['risk_category']} Risk"):
                        st.write(f"**Similarity Score:** {result['similarity']:.3f}")
                        st.write(f"**Risk Category:** {result['metadata']['risk_category']}")
                        st.write(f"**Claim Filed:** {'Yes' if result['metadata']['outcome'] == 1 else 'No'}")
                        st.write("**Profile:**")
                        st.write(result['profile'][:500] + "...")
    
    # Sidebar - Statistics
    st.sidebar.markdown("---")
    st.sidebar.header("📊 System Statistics")
    st.sidebar.metric("Total Customers", f"{len(historical_data):,}")
    st.sidebar.metric("Average Claim Rate", f"{historical_data['OUTCOME'].mean()*100:.1f}%")
    st.sidebar.metric("High Risk Customers", f"{(historical_data['RISK_CATEGORY']=='Very High').sum():,}")


def display_results(approval_status, claim_prob, risk_category, risk_score, 
                   customer_data, embedding_model, faiss_index, sentences, metadata):
    """Display analysis results"""
    
    st.markdown("---")
    st.header("📋 Underwriting Analysis Results")
    
    # Approval Decision
    approval_class = "approval-approved" if approval_status == "APPROVED" else "approval-declined"
    st.markdown(f"""
        <div class='{approval_class}'>
            <h2 style='margin:0;'>{"✅ APPROVED" if approval_status == "APPROVED" else "❌ DECLINED"}</h2>
            <p style='margin:5px 0 0 0; font-size: 1.1rem;'>
                Claim Probability: <strong>{claim_prob*100:.1f}%</strong> | 
                Approval Confidence: <strong>{(1-claim_prob)*100:.1f}%</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Category", risk_category)
    with col2:
        st.metric("Risk Score", f"{risk_score:.1f}")
    with col3:
        st.metric("Total Violations", customer_data['TOTAL_VIOLATIONS'])
    with col4:
        st.metric("Credit Score", f"{customer_data['CREDIT_SCORE']:.2f}")
    
    # Risk Gauge
    st.subheader("Risk Assessment Gauge")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = claim_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Claim Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "lightyellow"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Risk Factors
    st.subheader("⚠️ Key Risk Factors")
    risk_factors = []
    
    if customer_data['HAS_DUI'] == 1:
        risk_factors.append(("🚨 DUI on record", "critical"))
    if customer_data['TOTAL_VIOLATIONS'] >= 3:
        risk_factors.append((f"⚠️ {customer_data['TOTAL_VIOLATIONS']} total violations", "high"))
    if customer_data['YOUNG_INEXPERIENCED'] == 1:
        risk_factors.append(("⚠️ Young & inexperienced driver", "high"))
    if customer_data['LOW_CREDIT'] == 1:
        risk_factors.append(("⚠️ Low credit score", "medium"))
    if customer_data['HIGH_MILEAGE'] == 1:
        risk_factors.append(("⚠️ High annual mileage", "medium"))
    
    if risk_factors:
        for factor, level in risk_factors:
            st.warning(factor)
    else:
        st.success("✅ No significant risk factors identified")
    
    # Similar Customers
    st.subheader("👥 Similar Customer Profiles")
    query = f"risk score {risk_score:.0f}, violations {customer_data['TOTAL_VIOLATIONS']}"
    similar = search_similar_customers(query, embedding_model, faiss_index, sentences, metadata, top_k=3)
    
    cols = st.columns(3)
    for i, result in enumerate(similar):
        with cols[i]:
            st.markdown(f"""
                **Customer {result['metadata']['customer_id']}**
                - Risk: {result['metadata']['risk_category']}
                - Claimed: {'Yes' if result['metadata']['outcome'] == 1 else 'No'}
                - Similarity: {result['similarity']:.2f}
            """)
    
    # Recommendation
    st.subheader("💡 Underwriting Recommendation")
    recommendation = get_recommendation(claim_prob, risk_category)
    
    if "APPROVE" in recommendation:
        st.success(recommendation)
    elif "CONDITIONAL" in recommendation:
        st.warning(recommendation)
    else:
        st.error(recommendation)
    
    # Export report
    if st.button("📄 Generate Report", use_container_width=True):
        st.info("Report generation feature - Coming soon!")


if __name__ == "__main__":
    main()