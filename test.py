# ============================================================================
# FILE 4: Quick Test Script
# ============================================================================
"""
Save as: quick_test.py

Quick test to verify everything works
"""

import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def test_system():
    print("="*70)
    print("SYSTEM VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Data files
    print("\n[1/5] Checking data files...")
    try:
        df = pd.read_csv('..data/cleaned_insurance_data.csv')
        print(f"  ✓ Found {len(df)} customer records")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 2: ML Model
    print("\n[2/5] Checking ML model...")
    try:
        with open('..models/underwriting_model.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Model loaded: {data['metrics']['name']}")
        print(f"  ✓ ROC-AUC: {data['metrics']['roc_auc']:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 3: Embedding Model
    print("\n[3/5] Checking embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_emb = model.encode(["test sentence"])
        print(f"  ✓ Embedding dimension: {test_emb.shape[1]}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: FAISS Index
    print("\n[4/5] Checking FAISS index...")
    try:
        index = faiss.read_index('..models/faiss_index/customer.index')
        print(f"  ✓ Index loaded: {index.ntotal} vectors")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 5: Make a prediction
    print("\n[5/5] Testing prediction...")
    try:
        df_model = pd.read_csv('..data/modeling_ready_data.csv')
        customer = df_model.iloc[0]
        
        with open('..models/underwriting_model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            features = data['feature_names']
        
        X = pd.DataFrame([customer])[features]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        
        print(f"  ✓ Prediction: {'Claim' if pred == 1 else 'No Claim'}")
        print(f"  ✓ Probability: {prob*100:.1f}%")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(f"TESTS PASSED: {tests_passed}/{tests_total}")
    print("="*70)
    
    if tests_passed == tests_total:
        print("\n✅ ALL SYSTEMS OPERATIONAL!")
        print("\nYou can now:")
        print("  • Run: streamlit run app.py")
        print("  • Or: python cli_assistant.py")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Run: python underwriting_llm_system.py")

if __name__ == "__main__":
    test_system()