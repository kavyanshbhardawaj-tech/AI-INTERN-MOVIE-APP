import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
DATA = Path(__file__).parent / "data" / "movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA).fillna("")
    df["features"] = (df["genres"].str.replace("|", " ", regex=False) + " " + df["overview"])
    matrix = TfidfVectorizer(stop_words="english").fit_transform(df["features"])
    similarity = cosine_similarity(matrix)
    return df, similarity

df, similarity = load_data()
st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering using TF-IDF and cosine similarity.")
st.metric("Movies in catalogue", len(df))
st.info("This self-contained starter uses a curated 30-movie demonstration catalogue. Expand it with a larger, properly licensed dataset before making broad performance claims.")
selected = st.selectbox("Choose a movie you like", sorted(df["title"].tolist()))
n = st.slider("Number of recommendations", 3, 10, 5)
if st.button("Recommend movies", type="primary"):
    idx = df.index[df["title"] == selected][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted([x for x in scores if x[0] != idx], key=lambda x: x[1], reverse=True)[:n]
    st.subheader(f"Because you selected: {selected}")
    for rank, (i, score) in enumerate(scores, 1):
        row = df.iloc[i]
        with st.container(border=True):
            st.markdown(f"**{rank}. {row['title']}**")
            st.caption(f"Genres: {row['genres']}  ·  Similarity score: {score:.3f}")
            st.write(row["overview"])
st.subheader("Recommendation method")
st.markdown("- **Content-based filtering:** compares movie descriptions and genres.\n- **TF-IDF:** represents text as weighted numerical vectors.\n- **Cosine similarity:** ranks movies by the angle-based similarity of their vectors.")
st.caption("Similarity is based only on the included text features; it is not a prediction of personal taste or movie quality.")
