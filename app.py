import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    ratings = pd.read_csv('ratings_sample.csv')
    movies  = pd.read_csv('movies.csv')
    summary = pd.read_csv('model_summary.csv')
    return ratings, movies, summary

svd     = load_model()
df, movies, summary = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_top_k_recs(user_id, k=10):
    """Return top-k recommended movies for a user."""
    rated_movies = set(df[df['user_id'] == user_id]['movie_id'].tolist())
    all_movies   = df['movie_id'].unique().tolist()
    unrated      = [mid for mid in all_movies if mid not in rated_movies]

    preds = [(mid, svd.predict(user_id, mid).est) for mid in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_k = preds[:k]

    results = []
    for mid, score in top_k:
        title_row = movies[movies['movie_id'] == mid]
        title = title_row['title'].values[0] if len(title_row) > 0 else f'Movie {mid}'
        year  = title_row['year'].values[0]  if len(title_row) > 0 else 'N/A'
        results.append({'Title': title, 'Year': year, 'Predicted Rating': round(score, 2)})

    return pd.DataFrame(results)

def get_user_history(user_id):
    """Return movies rated by a user with titles."""
    user_ratings = df[df['user_id'] == user_id].merge(movies, on='movie_id')
    return user_ratings[['title', 'year', 'rating']].sort_values('rating', ascending=False)

def get_similar_movies(movie_id, k=10):
    """
    Find similar movies using item-item collaborative filtering.
    Computes cosine similarity between the target movie's latent factors
    and all other movies in the SVD model.
    """
    try:
        inner_iid = svd.trainset.to_inner_iid(movie_id)
    except ValueError:
        return pd.DataFrame()

    target_factors = svd.qi[inner_iid]  # latent factor vector for target movie

    scores = []
    for iid in range(svd.trainset.n_items):
        if iid == inner_iid:
            continue
        other_factors = svd.qi[iid]
        # Cosine similarity
        sim = np.dot(target_factors, other_factors) / (
            np.linalg.norm(target_factors) * np.linalg.norm(other_factors) + 1e-9
        )
        raw_iid = svd.trainset.to_raw_iid(iid)
        scores.append((raw_iid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = scores[:k]

    results = []
    for mid, sim in top_k:
        title_row = movies[movies['movie_id'] == mid]
        title = title_row['title'].values[0] if len(title_row) > 0 else f'Movie {mid}'
        year  = title_row['year'].values[0]  if len(title_row) > 0 else 'N/A'
        results.append({'Title': title, 'Year': year, 'Similarity Score': round(float(sim), 4)})

    return pd.DataFrame(results)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("🎬 Netflix Recommender")
st.sidebar.markdown("**Cult Open Projects 2026**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "👤 User Recommendations", "🎥 Similar Movies", "📊 Model Performance"]
)

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🎬 Netflix Recommendation System")
    st.markdown("### AI-powered personalized content discovery using the Netflix Prize Dataset")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings",  f"{len(df):,}")
    col2.metric("Unique Users",   f"{df['user_id'].nunique():,}")
    col3.metric("Unique Movies",  f"{df['movie_id'].nunique():,}")
    sparsity = 1 - len(df) / (df['user_id'].nunique() * df['movie_id'].nunique())
    col4.metric("Sparsity", f"{sparsity:.2%}")

    st.markdown("---")
    st.subheader("📈 Rating Distribution")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    rating_counts = df['rating'].value_counts().sort_index()
    axes[0].bar(rating_counts.index, rating_counts.values, color='#E50914', edgecolor='black')
    axes[0].set_title('Rating Distribution')
    axes[0].set_xlabel('Stars')
    axes[0].set_ylabel('Count')

    ratings_per_user = df.groupby('user_id').size()
    axes[1].hist(ratings_per_user, bins=40, color='#221F1F', edgecolor='white')
    axes[1].set_title('Ratings per User')
    axes[1].set_xlabel('Number of Ratings')
    axes[1].set_ylabel('Number of Users')
    axes[1].set_yscale('log')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🏆 Top 10 Most Rated Movies")
    top_movies = (
        df.groupby('movie_id')
        .agg(num_ratings=('rating', 'count'), avg_rating=('rating', 'mean'))
        .reset_index()
        .merge(movies, on='movie_id')
        .sort_values('num_ratings', ascending=False)
        .head(10)[['title', 'year', 'num_ratings', 'avg_rating']]
        .rename(columns={'title': 'Title', 'year': 'Year',
                         'num_ratings': 'No. of Ratings', 'avg_rating': 'Avg Rating'})
    )
    top_movies['Avg Rating'] = top_movies['Avg Rating'].round(2)
    st.dataframe(top_movies, use_container_width=True, hide_index=True)

# ── USER RECOMMENDATIONS ──────────────────────────────────────────────────────
elif page == "👤 User Recommendations":
    st.title("👤 Personalized Recommendations")
    st.markdown("Enter a User ID to see their taste profile and top recommendations.")
    st.markdown("---")

    # Sample user IDs for convenience
    sample_users = df['user_id'].value_counts().head(20).index.tolist()
    st.markdown(f"**Sample User IDs to try:** `{'`, `'.join(map(str, sample_users[:5]))}`")

    user_input = st.text_input("Enter User ID", placeholder="e.g. 1933317")

    if user_input:
        try:
            user_id = int(user_input)

            if user_id not in df['user_id'].values:
                st.error("User ID not found in dataset. Try one from the sample list above.")
            else:
                col1, col2 = st.columns(2)

                # ── Taste Profile ──
                with col1:
                    st.subheader("🎭 Taste Profile")
                    history = get_user_history(user_id)
                    total   = len(history)
                    avg_r   = history['rating'].mean()
                    loved   = len(history[history['rating'] >= 4])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Movies Rated", total)
                    m2.metric("Avg Rating Given", f"{avg_r:.2f}")
                    m3.metric("Loved (≥4⭐)", loved)

                    st.markdown("**Top Rated Movies:**")
                    top_rated = history.head(8)[['title', 'rating']].rename(
                        columns={'title': 'Title', 'rating': 'Rating'}
                    )
                    st.dataframe(top_rated, use_container_width=True, hide_index=True)

                # ── Recommendations ──
                with col2:
                    st.subheader("🎯 Top 10 Recommendations")
                    with st.spinner("Generating recommendations..."):
                        recs = get_top_k_recs(user_id, k=10)

                    if recs.empty:
                        st.warning("Not enough data to generate recommendations for this user.")
                    else:
                        # Color code by predicted rating
                        def highlight_score(val):
                            if val >= 4.0:
                                return 'background-color: #d4edda'
                            elif val >= 3.5:
                                return 'background-color: #fff3cd'
                            else:
                                return 'background-color: #f8d7da'

                        try:
                            styled = recs.style.map(
                                highlight_score, subset=['Predicted Rating']
                            )
                        except AttributeError:
                            styled = recs.style.applymap(
                                highlight_score, subset=['Predicted Rating']
                            )
                        st.dataframe(styled, use_container_width=True, hide_index=True)
                        st.caption("🟢 ≥4.0   🟡 3.5–4.0   🔴 <3.5")

                        # Explainability note
                        st.markdown("---")
                        st.markdown("**💡 Why these recommendations?**")
                        st.info(
                            f"These movies were selected because users with similar "
                            f"rating patterns to User {user_id} also enjoyed them. "
                            f"The SVD model identified latent preference factors "
                            f"(e.g. genre affinity, era preferences) from {total} rated movies."
                        )

        except ValueError:
            st.error("Please enter a valid numeric User ID.")

# ── SIMILAR MOVIES ────────────────────────────────────────────────────────────
elif page == "🎥 Similar Movies":
    st.title("🎥 Find Similar Movies")
    st.markdown("Enter a movie name to find similar titles based on latent factor similarity.")
    st.markdown("---")

    movie_input = st.text_input("Search Movie Title", placeholder="e.g. Lilo and Stitch")

    if movie_input:
        matches = movies[movies['title'].str.contains(movie_input, case=False, na=False)]

        if matches.empty:
            st.error("No movies found with that title. Try a different search.")
        else:
            st.markdown(f"**Found {len(matches)} match(es):**")
            selected_title = st.selectbox(
                "Select a movie",
                matches['title'].tolist()
            )

            selected_movie = movies[movies['title'] == selected_title].iloc[0]
            movie_id = selected_movie['movie_id']

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Selected Movie:**")
                st.markdown(f"🎬 **{selected_title}**")
                st.markdown(f"📅 Year: `{selected_movie['year']}`")

                movie_ratings = df[df['movie_id'] == movie_id]['rating']
                if len(movie_ratings) > 0:
                    st.markdown(f"⭐ Avg Rating: `{movie_ratings.mean():.2f}`")
                    st.markdown(f"📊 Total Ratings: `{len(movie_ratings):,}`")

            with col2:
                st.markdown("**🔍 Similar Movies (by latent factor similarity):**")
                with st.spinner("Finding similar movies..."):
                    similar = get_similar_movies(movie_id, k=10)

                if similar.empty:
                    st.warning("This movie doesn't have enough rating data for similarity analysis.")
                else:
                    st.dataframe(similar, use_container_width=True, hide_index=True)
                    st.caption(
                        "Similarity computed using cosine distance between SVD latent factors — "
                        "movies whose viewing patterns correlate with this title."
                    )

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Evaluation results for SVD vs User-Based Collaborative Filtering.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evaluation Metrics")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Metric Definitions:**")
        st.markdown("- **RMSE** — Root Mean Squared Error on held-out ratings (lower = better)")
        st.markdown("- **MAE** — Mean Absolute Error (lower = better)")
        st.markdown("- **MAP@10** — Mean Average Precision at 10 (higher = better, relevance threshold = 3.5⭐)")

    with col2:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        models = summary['Model'].tolist()
        rmses  = summary['RMSE'].tolist()
        maps   = summary['MAP@10'].tolist()

        axes[0].bar(models, rmses, color=['#E50914', '#221F1F'])
        axes[0].set_title('RMSE (lower is better)')
        axes[0].set_ylabel('RMSE')
        for i, v in enumerate(rmses):
            axes[0].text(i, v + 0.01, str(v), ha='center', fontweight='bold')

        axes[1].bar(models, maps, color=['#E50914', '#221F1F'])
        axes[1].set_title('MAP@10 (higher is better)')
        axes[1].set_ylabel('MAP@10')
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(maps):
            axes[1].text(i, v + 0.01, str(v), ha='center', fontweight='bold')

        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔍 Methodology")
    st.markdown("""
    | Step | Detail |
    |---|---|
    | **Dataset** | Netflix Prize Dataset (sampled 500K ratings from File 1) |
    | **Filtering** | Users with ≥10 ratings, Movies with ≥20 ratings |
    | **Train-Test Split** | 80% train / 20% test (random, seed=42) |
    | **Relevance Threshold** | Rating ≥ 3.5 considered relevant for MAP@10 |
    | **SVD Config** | 100 latent factors, 20 epochs, lr=0.005, reg=0.02 |
    | **CF Config** | KNNBasic, k=40 neighbors, cosine similarity, user-based |
    """)
