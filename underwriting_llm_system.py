"""
COMPLETE LLM-POWERED UNDERWRITING ASSISTANT
===============================================
Step 1: Train Approval Prediction Model
Step 2: Transform Dataset to Sentences (RAG)
Step 3: Embed Sentences
Step 4: Build FAISS Vector Store
Step 5: Deploy Interactive LLM Assistant
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# NLP & Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# STEP 1: TRAIN APPROVAL PREDICTION MODEL
# ============================================================================

class UnderwritingModel:
    """
    Predicts insurance approval and provides risk assessment
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_metrics = {}
        
    def load_data(self):
        """Load the cleaned modeling-ready data"""
        df = pd.read_csv('../data/modeling_ready_data.csv')
        
        # Separate features and target
        X = df.drop(['OUTCOME', 'ID'], axis=1)
        y = df['OUTCOME']
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, X, y):
        """Train and evaluate multiple models"""
        print("\n" + "="*70)
        print("TRAINING APPROVAL PREDICTION MODEL")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print('='*50)
            
            # Train
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"Test ROC-AUC: {roc_auc:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Claim', 'Claim']))
            
            # Track best model
            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = name
                self.model = model
                self.model_metrics = {
                    'name': name,
                    'roc_auc': roc_auc,
                    'cv_scores': cv_scores.tolist(),
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model_name} (ROC-AUC: {best_score:.4f})")
        print('='*70)
        
        # Plot confusion matrix and ROC curve
        self._plot_model_performance()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance(X.columns)
        
        return self.model
    
    def _plot_model_performance(self):
        """Plot confusion matrix and ROC curve"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(self.model_metrics['y_test'], self.model_metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                    xticklabels=['No Claim', 'Claim'],
                    yticklabels=['No Claim', 'Claim'])
        axes[0].set_title(f"Confusion Matrix - {self.model_metrics['name']}")
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.model_metrics['y_test'], self.model_metrics['y_pred_proba'])
        axes[1].plot(fpr, tpr, label=f"ROC (AUC = {self.model_metrics['roc_auc']:.3f})", linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../outputs/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Model performance plots saved")
    
    def _plot_feature_importance(self, feature_names):
        """Plot top 15 feature importances"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('../outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Feature importance plot saved")
    
    def predict(self, X):
        """Predict approval and get probability"""
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]
        return pred, proba
    
    def save_model(self, filepath='../models/underwriting_model.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics
            }, f)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='../models/underwriting_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model_obj = UnderwritingModel()
        model_obj.model = data['model']
        model_obj.feature_names = data['feature_names']
        model_obj.model_metrics = data['metrics']
        
        print(f"✓ Model loaded from {filepath}")
        return model_obj


# ============================================================================
# STEP 2: TRANSFORM DATASET TO SENTENCES (RAG)
# ============================================================================

class CustomerProfileTransformer:
    """
    Transforms structured customer data into natural language sentences for RAG
    """
    
    def __init__(self):
        self.customer_sentences = []
    
    def transform_to_sentences(self, df_path='../data/cleaned_insurance_data.csv'):
        """Convert each customer record to descriptive sentences"""
        print("\n" + "="*70)
        print("STEP 2: TRANSFORMING DATA TO SENTENCES")
        print("="*70)
        
        df = pd.read_csv(df_path)
        
        sentences = []
        metadata = []
        
        for idx, row in df.iterrows():
            # Create comprehensive customer profile
            profile = self._create_customer_profile(row)
            
            sentences.append(profile)
            metadata.append({
                'customer_id': row['ID'],
                'risk_category': row['RISK_CATEGORY'],
                'risk_score': row['RISK_SCORE'],
                'outcome': row['OUTCOME']
            })
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} customers...")
        
        print(f"\n✓ Transformed {len(sentences)} customer records to sentences")
        
        # Save
        self.customer_sentences = sentences
        self._save_sentences(sentences, metadata)
        
        return sentences, metadata
    
    def _create_customer_profile(self, row):
        """Create natural language profile for a customer"""
        
        # Basic demographics
        profile = f"Customer ID {row['ID']} is a {row['AGE']} year old {row['GENDER']} "
        profile += f"from the {row['RACE']} group with {row['DRIVING_EXPERIENCE']} of driving experience. "
        
        # Education and financial
        profile += f"They have {row['EDUCATION']} education and fall into the {row['INCOME']} income bracket. "
        profile += f"Their credit score is {row['CREDIT_SCORE']:.2f}. "
        
        # Vehicle information
        profile += f"They {'own' if row['VEHICLE_OWNERSHIP'] == 1 else 'do not own'} their vehicle, "
        profile += f"which is a {row['VEHICLE_TYPE']} manufactured {row['VEHICLE_YEAR']}. "
        profile += f"They drive approximately {row['ANNUAL_MILEAGE']:.0f} miles annually. "
        
        # Personal status
        marital = "married" if row['MARRIED'] == 1 else "not married"
        children = f"with {int(row['CHILDREN'])} {'child' if row['CHILDREN'] == 1 else 'children'}" if row['CHILDREN'] > 0 else "with no children"
        profile += f"This customer is {marital} {children}. "
        
        # Driving record
        if row['TOTAL_VIOLATIONS'] == 0:
            profile += "They have a clean driving record with no violations or accidents. "
        else:
            violations = []
            if row['SPEEDING_VIOLATIONS'] > 0:
                violations.append(f"{int(row['SPEEDING_VIOLATIONS'])} speeding violation(s)")
            if row['DUIS'] > 0:
                violations.append(f"{int(row['DUIS'])} DUI(s)")
            if row['PAST_ACCIDENTS'] > 0:
                violations.append(f"{int(row['PAST_ACCIDENTS'])} past accident(s)")
            
            profile += f"Their driving record includes: {', '.join(violations)}. "
        
        # Risk assessment
        profile += f"Risk Assessment: This customer is classified as {row['RISK_CATEGORY']} risk "
        profile += f"with a risk score of {row['RISK_SCORE']:.2f}. "
        
        # Key risk factors
        risk_factors = []
        if row['YOUNG_INEXPERIENCED'] == 1:
            risk_factors.append("young and inexperienced driver")
        if row['HAS_DUI'] == 1:
            risk_factors.append("DUI history")
        if row['LOW_CREDIT'] == 1:
            risk_factors.append("low credit score")
        if row['HIGH_MILEAGE'] == 1:
            risk_factors.append("high annual mileage")
        
        if risk_factors:
            profile += f"Key risk factors include: {', '.join(risk_factors)}. "
        else:
            profile += "No significant risk factors identified. "
        
        # Outcome
        profile += f"Historical outcome: This customer {'filed a claim' if row['OUTCOME'] == 1 else 'did not file a claim'}."
        
        return profile
    
    def _save_sentences(self, sentences, metadata):
        """Save sentences and metadata"""
        Path('../data/rag_data').mkdir(parents=True, exist_ok=True)
        
        # Save sentences
        with open('../data/rag_data/customer_sentences.txt', 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n\n')
        
        # Save metadata
        with open('../data/rag_data/customer_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Sentences and metadata saved")


# ============================================================================
# STEP 3 & 4: EMBED SENTENCES AND BUILD FAISS INDEX
# ============================================================================

class UnderwritingRAG:
    """
    RAG system for underwriting using sentence embeddings and FAISS
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model"""
        print(f"\n{'='*70}")
        print("STEP 3 & 4: EMBEDDING & BUILDING FAISS INDEX")
        print('='*70)
        print(f"Loading embedding model: {model_name}...")
        
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.sentences = []
        self.metadata = []
        self.dimension = 384  # MiniLM dimension
        
        print("✓ Embedding model loaded")
    
    def build_index(self, sentences, metadata):
        """Embed sentences and build FAISS index"""
        print(f"\nEmbedding {len(sentences)} customer profiles...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            sentences, 
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.dimension = embeddings.shape[1]
        
        # Using IndexFlatL2 for exact search (good for <1M vectors)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.sentences = sentences
        self.metadata = metadata
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
        
        return embeddings
    
    def search_similar_customers(self, query, top_k=5):
        """Search for similar customers based on query"""
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'customer_profile': self.sentences[idx],
                'metadata': self.metadata[idx],
                'similarity_score': float(1 / (1 + dist))  # Convert distance to similarity
            })
        
        return results
    
    def save_index(self, index_path='../models/faiss_index'):
        """Save FAISS index and metadata"""
        Path(index_path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f'{index_path}/customer.index')
        
        # Save sentences and metadata
        with open(f'{index_path}/sentences.pkl', 'wb') as f:
            pickle.dump(self.sentences, f)
        
        with open(f'{index_path}/metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✓ FAISS index saved to {index_path}")
    
    @staticmethod
    def load_index(index_path='../models/faiss_index', model_name='all-MiniLM-L6-v2'):
        """Load FAISS index and metadata"""
        rag = UnderwritingRAG(model_name)
        
        # Load FAISS index
        rag.index = faiss.read_index(f'{index_path}/customer.index')
        
        # Load sentences and metadata
        with open(f'{index_path}/sentences.pkl', 'rb') as f:
            rag.sentences = pickle.load(f)
        
        with open(f'{index_path}/metadata.pkl', 'rb') as f:
            rag.metadata = pickle.load(f)
        
        print(f"✓ FAISS index loaded from {index_path}")
        return rag


# ============================================================================
# STEP 5: LLM UNDERWRITING ASSISTANT
# ============================================================================

class UnderwritingAssistant:
    """
    Complete LLM-powered underwriting assistant
    Combines ML model predictions with RAG retrieval
    """
    
    def __init__(self, model_path='../models/underwriting_model.pkl',
                 index_path='../models/faiss_index'):
        """Initialize assistant with ML model and RAG system"""
        print("\n" + "="*70)
        print("INITIALIZING UNDERWRITING ASSISTANT")
        print("="*70)
        
        # Load ML model
        self.ml_model = UnderwritingModel.load_model(model_path)
        
        # Load RAG system
        self.rag = UnderwritingRAG.load_index(index_path)
        
        print("\n✓ Underwriting Assistant Ready!")
    
    def analyze_customer(self, customer_data):
        """
        Complete customer analysis with:
        1. ML prediction
        2. Similar customer retrieval
        3. Natural language summary
        """
        print("\n" + "="*70)
        print("CUSTOMER UNDERWRITING ANALYSIS")
        print("="*70)
        
        # Step 1: ML Prediction
        X = pd.DataFrame([customer_data])[self.ml_model.feature_names]
        prediction, probability = self.ml_model.predict(X)
        
        claim_prob = probability[0]
        approval_status = "APPROVED" if claim_prob < 0.5 else "DECLINED"
        
        print(f"\n🎯 APPROVAL DECISION: {approval_status}")
        print(f"   Claim Probability: {claim_prob*100:.2f}%")
        print(f"   Approval Confidence: {(1-claim_prob)*100:.2f}%")
        
        # Step 2: Retrieve similar customers
        query = self._create_query_from_data(customer_data)
        similar_customers = self.rag.search_similar_customers(query, top_k=3)
        
        print(f"\n🔍 Similar Customer Profiles (Top 3):")
        for i, customer in enumerate(similar_customers, 1):
            print(f"\n   {i}. Customer {customer['metadata']['customer_id']}")
            print(f"      Risk: {customer['metadata']['risk_category']}")
            print(f"      Claim: {'Yes' if customer['metadata']['outcome'] == 1 else 'No'}")
            print(f"      Similarity: {customer['similarity_score']:.3f}")
        
        # Step 3: Generate summary
        summary = self._generate_summary(customer_data, claim_prob, approval_status, similar_customers)
        
        return {
            'approval_status': approval_status,
            'claim_probability': claim_prob,
            'similar_customers': similar_customers,
            'summary': summary
        }
    
    def _create_query_from_data(self, data):
        """Create search query from customer data"""
        query = f"{data.get('AGE_NUMERIC', 2)} years driving experience, "
        query += f"risk score around {data.get('RISK_SCORE', 10)}, "
        query += f"violations: {data.get('TOTAL_VIOLATIONS', 0)}"
        return query
    
    def _generate_summary(self, data, claim_prob, status, similar_customers):
        """Generate natural language summary"""
        summary = f"\n{'='*70}\n"
        summary += "UNDERWRITING SUMMARY\n"
        summary += f"{'='*70}\n\n"
        
        summary += f"Decision: {status}\n"
        summary += f"Claim Risk: {claim_prob*100:.1f}%\n\n"
        
        summary += "Key Factors:\n"
        
        # Risk factors
        if data.get('HAS_DUI', 0) == 1:
            summary += "  ⚠️  CRITICAL: DUI on record\n"
        
        if data.get('TOTAL_VIOLATIONS', 0) >= 3:
            summary += f"  ⚠️  HIGH: {data['TOTAL_VIOLATIONS']} total violations\n"
        
        if data.get('YOUNG_INEXPERIENCED', 0) == 1:
            summary += "  ⚠️  Young and inexperienced driver\n"
        
        if data.get('LOW_CREDIT', 0) == 1:
            summary += "  ⚠️  Low credit score\n"
        
        # Positive factors
        if data.get('TOTAL_VIOLATIONS', 0) == 0:
            summary += "  ✓  Clean driving record\n"
        
        if data.get('EXPERIENCE_NUMERIC', 1) >= 3:
            summary += "  ✓  Experienced driver\n"
        
        # Similar customers insight
        avg_claim_rate = np.mean([c['metadata']['outcome'] for c in similar_customers])
        summary += f"\nSimilar Customers: {avg_claim_rate*100:.0f}% filed claims\n"
        
        # Recommendation
        summary += "\nRecommendation:\n"
        if status == "APPROVED":
            summary += "  Proceed with standard underwriting process.\n"
        else:
            summary += "  Recommend manual review or higher premium tier.\n"
        
        summary += f"\n{'='*70}\n"
        
        print(summary)
        return summary


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete pipeline"""
    print("\n" + "="*70)
    print("UNDERWRITING LLM SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    # STEP 1: Train ML Model
    print("\n[STEP 1/5] Training Approval Prediction Model...")
    ml_model = UnderwritingModel()
    X, y = ml_model.load_data()
    ml_model.train_model(X, y)
    ml_model.save_model()
    
    # STEP 2: Transform to Sentences
    print("\n[STEP 2/5] Transforming Data to Sentences...")
    transformer = CustomerProfileTransformer()
    sentences, metadata = transformer.transform_to_sentences()
    
    # STEP 3 & 4: Embed and Build FAISS
    print("\n[STEP 3-4/5] Embedding and Building FAISS Index...")
    rag = UnderwritingRAG()
    rag.build_index(sentences, metadata)
    rag.save_index()
    
    # STEP 5: Initialize Assistant
    print("\n[STEP 5/5] Initializing LLM Assistant...")    
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
    print("="*70)
    print("\nSystem Ready for Deployment! 🚀")
    print("\nNext Steps:")
    print("  1. Test with real customer queries")
    print("  2. Build web interface (Streamlit/Gradio)")
    print("  3. Integrate with existing systems")
    print("  4. Monitor and iterate")


if __name__ == "__main__":
    main()