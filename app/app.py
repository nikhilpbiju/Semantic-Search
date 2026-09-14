import sys
import os

# Fix import path so src/ works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.search import search

st.set_page_config(page_title="Semantic Search", layout="wide")

st.title("🔎 Semantic Product Discovery Engine")

st.markdown("Search products using meaning + keywords (hybrid search)")

query = st.text_input("Search products...")

if query:
    results = search(query, k=5)

    st.subheader("Top Results")

    for _, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        st.write(row["description"])
        st.caption(f"Relevance Score: {row['final_score']:.4f}")
        st.markdown("---")