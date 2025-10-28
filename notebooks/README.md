
 # UnderwriteGPT: Retrieval-Augmented Risk Assessment (Colab workflow)

A simple, step-by-step guide to build a free, Colab-based prototype that retrieves similar past insurance policies and explains risk for a new case. Read like a short story: start with the data, clean it, turn rows into human sentences, make vectors, ask the system a question, and show the answers.

---

## Idea in one line

You give a new policy description; the system finds similar past policies and explains why the new case looks risky (or not), using actual past outcomes as evidence.

---

## Folder & file layout (what you will create)

```
underwritegpt/
│
├── data/
│   └── Insurance_claims_data.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_text_generation.ipynb
│   ├── 03_rag_retrieval.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── embeddings_faiss.index
│
├── requirements.txt
└── README.md
```

---

## Quick prerequisites (Colab)

1. Open a new Colab notebook.
2. Mount Google Drive if you want persistent storage:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install the small packages you will need (run once per session):

```bash
!pip install -q sentence-transformers faiss-cpu streamlit pyngrok pandas scikit-learn plotly
```

---

## Step 1 — Load and inspect the raw data (story: meet your data)

1. Load CSV into a dataframe.

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/underwritegpt/data/raw/motor_insurance.csv')
df.shape, df.head()
```

2. Quick checks:

```python
df.info()
df.isna().sum()
df.describe(include='all').T
```

Goal: understand missingness, obvious type errors, and which columns are categorical vs numeric.

---

## Step 2 — Clean the data (story: tidy the characters)

Do simple, safe cleaning so you can move fast.

1. Standardise column names:

```python
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
```

2. Convert yes/no to binary:

```python
binary_cols = ['is_esc','is_tpms','is_parking_sensors','is_parking_camera']  # example list
for c in binary_cols:
    df[c] = df[c].map({'Yes':1,'No':0}).fillna(0).astype(int)
```

3. Handle missing numeric values:

```python
num_cols = df.select_dtypes(include='number').columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
```

4. Handle missing categorical values:

```python
cat_cols = ['model','fuel_type','region_code']
for c in cat_cols:
    df[c] = df[c].fillna('unknown').astype(str)
```

5. Save cleaned data:

```python
df.to_csv('/content/drive/MyDrive/underwritegpt/outputs/cleaned_data.csv', index=False)
```

Tip: keep a copy of raw data untouched. Cleaning is iterative; document every transformation in your notebook.

---

## Step 3 — Create simple natural-language summaries (story: turn rows into mini case notes)

Each row becomes a 1–3 sentence summary. These are the “documents” for retrieval.

Example summary template:

```python
def make_summary(r):
    return (
        f"{r.customer_age}-year-old policyholder in region {r.region_code} with a "
        f"{r.vehicle_age}-year-old {r.fuel_type} {r.model}. "
        f"Safety: {r.airbags} airbags, ESC={'yes' if r.is_esc==1 else 'no'}. "
        f"Premium: {r.subscription_length} months. "
        f"Claim: {'Yes' if r.claim_status==1 else 'No'}."
    )

df['summary'] = df.apply(make_summary, axis=1)
df[['policy_id','summary','claim_status']].head()
```

Why: short readable text captures structured info and is excellent for embeddings.

---

## Step 4 — Generate embeddings (story: give each case a vector fingerprint)

Use a small, free sentence-transformer model in Colab.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')   # small and fast
texts = df['summary'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
```

Save embeddings if you want to reuse them:

```python
import numpy as np
np.save('/content/drive/MyDrive/underwritegpt/models/embeddings.npy', embeddings)
```

---

## Step 5 — Build a FAISS index (story: a fast memories table)

FAISS lets you find nearest neighbours quickly.

```python
import faiss
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)   # add all vectors
faiss.write_index(index, '/content/drive/MyDrive/underwritegpt/models/faiss_index.bin')
```

To load later:

```python
index = faiss.read_index('/content/drive/MyDrive/underwritegpt/models/faiss_index.bin')
```

---

## Step 6 — Query & retrieve similar cases (story: ask the memory)

1. Build a simple query string from a new case:

```python
query = "24-year-old, 8-year-old petrol Toyota, no ESC, urban region 101"
q_vec = model.encode([query])
k = 3
D, I = index.search(q_vec, k)
similar_cases = df.iloc[I[0]]
```

2. Inspect results:

```python
similar_cases[['policy_id','summary','claim_status']]
```

Interpretation: D gives distances; lower distance = more similar. Show the retrieved rows and their claim outcomes to explain risk.

---

## Step 7 — Produce a human-friendly explanation (story: the voice)

Combine retrieved cases into a short explanation template:

```python
def explain(query, similar_df, distances):
    positives = similar_df['claim_status'].sum()
    total = len(similar_df)
    rate = positives / total
    lines = [
        f"The top {total} similar past cases show {positives} claims ({rate:.0%}).",
        "Similar cases:",
    ]
    for i, row in similar_df.iterrows():
        lines.append(f"- {row.policy_id}: {row.summary} (Claim={row.claim_status})")
    return "\n".join(lines)

print(explain(query, similar_cases, D[0]))
```

Goal: short, evidence-based, and actionable phrasing for an underwriter.

---

## Step 8 — Minimal Streamlit interface (story: show the work)

Create `app/streamlit_app.py` (very simple) to collect input and display results.

```python
# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

@st.cache_resource
def load_resources():
    df = pd.read_csv('outputs/cleaned_data.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('models/faiss_index.bin')
    return df, model, index

df, model, index = load_resources()

st.title("UnderwriteGPT — Claim Similarity Explorer")
policy_text = st.text_input("Describe the new case (age, vehicle, region, safety features):")
k = st.slider("Number of similar cases", 1, 10, 3)

if policy_text:
    qvec = model.encode([policy_text])
    D, I = index.search(qvec, k)
    results = df.iloc[I[0]]
    st.subheader("Similar cases")
    st.table(results[['policy_id','summary','claim_status']])
    positives = results['claim_status'].sum()
    st.write(f"Claims among top {k}: {positives}/{k} ({positives/k:.0%})")
```

Run locally in Colab if you wish:

```bash
!streamlit run app/streamlit_app.py & npx localtunnel --port 8501
```

Colab will provide a temporary public URL.

---

## Step 9 — Notes on explainability & visuals

* Show similarity scores as a small bar chart (Plotly) so users see how similar each retrieved case is.
* Show the claim rate among retrieved cases as a simple percentage.
* Always present retrieved cases — the evidence is the explainability.

---

## Step 10 — Save, document, and push to GitHub

1. Keep your notebooks clean and commented.
2. Save the cleaned CSV and model files to Drive or the repo.
3. Push code and markdown to GitHub. Add a short demo GIF or screenshots to the repo README.

---

## Minimal evaluation ideas (keep it cheap)

* Simple baseline: how often does nearest neighbour claim_status predict the new case? Compute accuracy/precision.
* Use stratified holdout to simulate new cases.
* Record retrieval examples that went well and those that failed; this is valuable for improving summaries.

---

## Final advice — The story continues

Start small: clean a few hundred rows, make summaries, build the index, and ask the system a handful of questions. Each question teaches you how to adjust summaries, which fields matter, and which safety features reduce claim rates. Keep iterations fast and recorded in your notebooks.

I
