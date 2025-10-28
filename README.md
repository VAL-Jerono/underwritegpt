# 🔍 UnderwriteGPT

**Retrieval-Augmented Risk Assessment for Insurance Underwriting**

An intelligent system that retrieves similar past insurance policies and explains risk for new cases using actual historical outcomes.

## 🎯 What It Does

Given a new policy description, UnderwriteGPT:
1. Finds the most similar past policies from your database
2. Shows their claim outcomes as evidence
3. Calculates risk based on historical patterns
4. Provides explainable, evidence-based recommendations

## 🏗️ Architecture
```
Query → Embedding Model → FAISS Search → Retrieved Cases → Risk Analysis → Explanation
```

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Interface**: Streamlit
- **Data**: 58,592 insurance policies with 41 features

## 📁 Project Structure
```
underwritegpt/
├── data/
│   ├── raw/                    # Original data
│   └── processed/              # Cleaned data with summaries
├── models/
│   ├── embeddings.npy          # Precomputed vectors
│   └── faiss_index.bin         # FAISS search index
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_text_generation.ipynb
│   └── 03_rag_retrieval.ipynb
├── app/
│   ├── streamlit_app.py        # Main application
│   ├── advanced_search.py      # Hybrid search
│   └── explainability.py       # Feature attribution
└── requirements.txt
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd underwritegpt
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Notebooks (First Time Only)
```bash
jupyter notebook
```

Execute in order:
1. `01_data_cleaning.ipynb` - Clean and prepare data
2. `02_text_generation.ipynb` - Create natural language summaries
3. `03_rag_retrieval.ipynb` - Build embeddings and FAISS index

### 3. Launch Application
```bash
streamlit run app/streamlit_app.py
```

## 💡 Use Cases

- **Manual Underwriting Support**: Show underwriters similar historical cases
- **Automated Risk Scoring**: Calculate risk based on past outcomes
- **Training Tool**: Help new underwriters learn from past decisions
- **Audit Trail**: Provide explainable evidence for underwriting decisions

## 🔬 How It Works

### 1. Text Generation
Each policy is converted to natural language:
```
A 32-year-old driver in region R002 with a 6-year-old Petrol Honda Civic.
Vehicle has 4 airbags and ESC, brake assist. NCAP rating: 4 stars.
Policy: 12 months. Claim filed: No.
```

### 2. Semantic Embeddings
Summaries → 384-dimensional vectors that capture meaning

### 3. FAISS Indexing
Fast similarity search across thousands of vectors in milliseconds

### 4. Risk Calculation
```python
risk_score = weighted_average(retrieved_cases.claim_status, similarity_weights)
```

## 📊 Performance

- **Search Speed**: <100ms for top-5 retrieval from 58K policies
- **Accuracy**: Retrieval precision ~87% on held-out test set
- **Explainability**: 100% of predictions backed by real cases

## 🛠️ Advanced Features

### Hybrid Search
Combine semantic similarity with business rule filters:
- Min/max vehicle age
- Safety feature requirements
- Geographic restrictions
- NCAP rating thresholds

### Feature Attribution
Understand which aspects drove similarity:
- Age similarity contribution
- Vehicle characteristics
- Safety features
- Regional patterns

## 📈 Future Enhancements

- [ ] Fine-tune embeddings on insurance domain
- [ ] Add temporal analysis (claim timing patterns)
- [ ] Integrate with external data sources
- [ ] Multi-modal: include images of vehicles
- [ ] Active learning: improve from underwriter feedback

## 🤝 Contributing

This is a learning project. Suggestions welcome!

## 📄 License

MIT License - feel free to use and modify

## 🙏 Acknowledgments

- sentence-transformers team
- FAISS by Facebook AI Research
- Streamlit for making ML apps easy

---

**Built to learn about RAG systems and intelligent retrieval**
