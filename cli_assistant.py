# ============================================================================
# FILE 2: cli_assistant.py - Command Line Interface
# ============================================================================
"""

Simple CLI for quick underwriting checks
Run with: python cli_assistant.py
"""

import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class CLIUnderwritingAssistant:
    """Command-line underwriting assistant"""
    
    def __init__(self):
        print("Loading models...")
        
        # Load ML model
        with open('../models/underwriting_model.pkl', 'rb') as f:
            data = pickle.load(f)
            self.ml_model = data['model']
            self.feature_names = data['feature_names']
        
        # Load RAG
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index('../models/faiss_index/customer.index')
        
        with open('../models/faiss_index/sentences.pkl', 'rb') as f:
            self.sentences = pickle.load(f)
        with open('../models/faiss_index/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print("✓ Models loaded!\n")
    
    def quick_analysis(self, customer_id=None):
        """Quick analysis of a customer from historical data"""
        df = pd.read_csv('../data/modeling_ready_data.csv')
        
        if customer_id:
            customer = df[df['ID'] == customer_id].iloc[0]
        else:
            # Random customer
            customer = df.sample(1).iloc[0]
        
        X = pd.DataFrame([customer])[self.feature_names]
        pred = self.ml_model.predict(X)[0]
        prob = self.ml_model.predict_proba(X)[0][1]
        
        print("="*70)
        print(f"CUSTOMER ID: {int(customer['ID'])}")
        print("="*70)
        
        status = "✅ APPROVED" if prob < 0.5 else "❌ DECLINED"
        print(f"\nDecision: {status}")
        print(f"Claim Probability: {prob*100:.1f}%")
        print(f"Risk Score: {customer['RISK_SCORE']:.1f}")
        print(f"Total Violations: {int(customer['TOTAL_VIOLATIONS'])}")
        
        # Similar customers
        query = f"risk score {customer['RISK_SCORE']}"
        query_emb = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_emb.astype('float32'), 3)
        
        print(f"\nSimilar Customers:")
        for idx in indices[0]:
            meta = self.metadata[idx]
            print(f"  - Customer {meta['customer_id']}: {meta['risk_category']} risk, "
                  f"{'Claimed' if meta['outcome'] == 1 else 'No claim'}")
        
        print("\n" + "="*70)
    
    def natural_query(self):
        """Natural language search"""
        query = input("\nEnter your search query: ")
        
        query_emb = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_emb.astype('float32'), 5)
        
        print("\nTop 5 Similar Customers:")
        print("="*70)
        
        for i, idx in enumerate(indices[0], 1):
            meta = self.metadata[idx]
            print(f"\n{i}. Customer {meta['customer_id']}")
            print(f"   Risk: {meta['risk_category']}")
            print(f"   Outcome: {'Claim' if meta['outcome'] == 1 else 'No Claim'}")
            print(f"   Profile: {self.sentences[idx][:200]}...")
    
    def run(self):
        """Main CLI loop"""
        while True:
            print("\n" + "="*70)
            print("UNDERWRITING ASSISTANT - CLI")
            print("="*70)
            print("\nOptions:")
            print("  1. Quick Analysis (Random Customer)")
            print("  2. Analyze Specific Customer ID")
            print("  3. Natural Language Search")
            print("  4. Exit")
            
            choice = input("\nSelect option (1-4): ")
            
            if choice == '1':
                self.quick_analysis()
            elif choice == '2':
                cid = int(input("Enter Customer ID: "))
                self.quick_analysis(cid)
            elif choice == '3':
                self.natural_query()
            elif choice == '4':
                print("\nGoodbye!")
                break
            else:
                print("Invalid option!")

if __name__ == "__main__":
    assistant = CLIUnderwritingAssistant()
    assistant.run()