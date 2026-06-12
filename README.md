# 🎬 Netflix Recommendation System
### Cult Open Projects 2026 — Personalized Content Discovery

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://netflix-recommender-frv5acygcv6v9ffixmjtvl.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Dashboard
**[netflix-recommender-frv5acygcv6v9ffixmjtvl.streamlit.app](https://netflix-recommender-frv5acygcv6v9ffixmjtvl.streamlit.app)**

---

## 📌 Overview

This project builds a personalized movie recommendation system using the **Netflix Prize Dataset** — one of the most influential datasets in recommendation system research.

We implement and compare two collaborative filtering approaches:
- **SVD (Matrix Factorization)** — learns latent user and movie factor vectors
- **User-Based Collaborative Filtering** — recommends based on similar users

The system is evaluated on **RMSE** (rating prediction accuracy) and **MAP@10** (recommendation ranking quality), and deployed as a live interactive Streamlit dashboard.

---

## 📊 Results

| Model | RMSE | MAE | MAP@10 |
|---|---|---|---|
| SVD (Matrix Factorization) | **0.9677** | **0.7663** | **0.8514** |
| User-Based CF | 1.0977 | 0.8820 | 0.8441 |

> Relevance threshold: actual rating ≥ 3.5 stars  
> Train/Test split: 80% / 20% (random, seed=42)

---

## 🗂️ Repository Structure

```
netflix-recommender/
│
├──netflix_recommendation.ipynb
| app.py                  # Streamlit dashboard (4 pages)
├── svd_model.pkl           # Trained SVD model
├── ratings_sample.csv      # Sampled ratings (50K rows)
├── movies.csv              # Movie metadata (titles + years)
├── model_summary.csv       # RMSE / MAE / MAP@10 results
├── requirements.txt       # Python dependencies
└── README.md

```

> **Note:** The full training notebook (`netflix_recommendation.ipynb`) is run in Google Colab.  
> Dataset: [Netflix Prize Data on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

---

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/Prathamesh-Barve/netflix-recommender.git
cd netflix-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard locally
```bash
streamlit run app.py
```

---

## 🔁 Reproducing Results

### Step 1 — Download the dataset
Go to [Kaggle Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and download the dataset. Place the extracted files in a `./netflix_data/` folder.

### Step 2 — Run the notebook in Google Colab
Open `netflix_recommendation.ipynb` in Google Colab (File → Upload Notebook). The notebook covers:

| Cell | Description |
|---|---|
| 1 | Install dependencies |
| 2 | Download dataset via Kaggle API |
| 3 | Parse raw Netflix `.txt` format |
| 4 | Sample & filter data (500K rows) |
| 5–7 | EDA: rating distribution, user activity, top movies |
| 8 | Prepare data for Surprise library (80/20 split) |
| 9 | Train SVD model → compute RMSE |
| 10 | Train User-Based CF → compute RMSE |
| 11 | Model comparison chart |
| 12 | MAP@10 computation |
| 13 | Generate Top-10 recommendations for sample users |
| 14 | Final summary table |

### Step 3 — Save outputs
Run the final save cell in the notebook to export:
```python
import pickle
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)
df.sample(n=50_000).to_csv('ratings_sample.csv', index=False)
movies.to_csv('movies.csv', index=False)
summary.to_csv('model_summary.csv', index=False)
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | Dataset stats, rating distribution, user activity histogram, top-10 movies |
| 👤 User Recommendations | Enter User ID → taste profile + SVD Top-10 recs with confidence color coding |
| 🎥 Similar Movies | Search movie title → 10 similar movies via SVD latent factor cosine similarity |
| 📊 Model Performance | RMSE & MAP@10 charts, full methodology, metric definitions |

---

## 🧠 Methodology

### Data Pipeline
1. **Parse** — custom parser handles Netflix's `movie_id:` header format
2. **Filter** — users with ≥10 ratings, movies with ≥20 ratings
3. **Sample** — 500K rows from `combined_data_1.txt` (Colab memory constraints)
4. **Split** — 80% train / 20% test (random, seed=42)

### Models
**SVD** decomposes the user-movie matrix R ≈ P × Qᵀ into latent factors:
- 100 latent factors, 20 epochs, lr=0.005, reg=0.02
- Handles sparsity well — learns global patterns across all users

**User-Based CF** finds k=40 most similar users via cosine similarity:
- Intuitive & explainable — "users like you also watched..."
- Computationally expensive at scale

### Evaluation
- **RMSE** — average rating prediction error (lower = better)
- **MAP@10** — mean average precision at 10 (higher = better)
- A movie is **relevant** if actual rating ≥ 3.5 stars

---

## 🔍 Explainability

The Similar Movies feature uses **cosine similarity on SVD latent vectors** (qi vectors) — no additional training needed. Movies that occupy similar positions in the learned latent space are surfaced as recommendations, providing an interpretable basis for suggestions.

---

## 🚀 Optional Tasks Completed

| Task | Status |
|---|---|
| A. Explainable Recommendations | ✅ Latent factor cosine similarity |
| C. Interactive Dashboard | ✅ 4-page Streamlit app |
| E. Deployment | ✅ Live on Streamlit Cloud |

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-surprise
matplotlib
```

---

## 🔮 Future Improvements

- Train on full 100M ratings using distributed computing
- Implement **SVD++** for implicit feedback handling
- Add **Neural Collaborative Filtering (NCF)** for non-linear interactions
- Build a **hybrid system** combining SVD with content features (genre, year)
- Add **A/B testing** framework for production deployment
