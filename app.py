import re
import textwrap
import pandas as pd
import streamlit as st
from openai import OpenAI
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Disinformation Narrative Analyzer", layout="wide")
st.title("Disinformation Narrative Analysis")

# --- API Key Input ---
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.stop()
client = OpenAI(api_key=openai_api_key)

# --- Input Selection ---
input_option = st.radio(
    "Choose input method:",
    ("Upload a .txt file", "Paste text manually")
)

article_text = ""

if input_option == "Upload a .txt file":
    uploaded_file = st.file_uploader("Upload an article (.txt) for analysis", type=["txt"])
    if uploaded_file is not None:
        article_text = uploaded_file.read().decode("utf-8")
    else:
        st.info("Please upload a `.txt` file to begin.")
        st.stop()
else:
    article_text = st.text_area("Paste your article text here:", height=300)
    if not article_text.strip():
        st.info("Please paste text to begin.")
        st.stop()

# --- Preprocessing ---
def preprocess_text(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    return re.split(r"(?<=[.!?])\s+", text)

def chunk_text(text: str, max_chunk_size=3000):
    sentences = preprocess_text(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- LLM Analysis ---
@st.cache_data(show_spinner=False)
def analyze_disinformation_llm(article_text: str, model="gpt-4o-mini"):
    chunks = chunk_text(article_text)
    all_results = []
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"Analyzing chunk {i}/{len(chunks)}...")
        prompt = f"""
        You are a disinformation analyst.
        Identify any disinformation/misinformation/influence narratives.
        For each:
        Disinformation Narrative Identified:

        Supporting Excerpts:

        Analysis:

        Text:
        {textwrap.shorten(chunk, width=4000, placeholder="...")}
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in disinformation detection."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        chunk_result = response.choices[0].message.content.replace("**", "")
        all_results.append(chunk_result)
    return all_results

# --- Semantic Grouping ---
def semantic_group_narratives(narratives, similarity_threshold=0.75):
    if len(narratives) <= 1:
        return {0: narratives}
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(narratives)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings)
    clustered = defaultdict(list)
    for label, text in zip(labels, narratives):
        clustered[label].append(text)
    # Use first narrative in each cluster as representative
    merged = {}
    for label, group in clustered.items():
        merged_label = group[0]
        merged[merged_label] = len(group)
    return merged

# --- Run Analysis ---
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        results = analyze_disinformation_llm(article_text)
        st.success("Analysis complete!")

        # Display raw chunk results with wrapped text
        st.subheader("Per-Chunk Analysis")
        for i, res in enumerate(results):
            with st.expander(f"Chunk {i+1}", expanded=False):
                wrapped_text = textwrap.fill(res.strip(), width=120)
                st.markdown(f"<pre style='white-space: pre-wrap;'>{wrapped_text}</pre>", unsafe_allow_html=True)

        # Extract narratives per chunk
        narrative_chunks = defaultdict(list)  # narrative -> list of chunk indices
        all_narratives = []
        for idx, r in enumerate(results):
            matches = re.findall(r"Disinformation Narrative Identified:\s*(.+)", r)
            matches = [m.strip() for m in matches if len(m.strip()) > 2]
            for m in matches:
                narrative_chunks[m].append(idx+1)
            all_narratives.extend(matches)

        if all_narratives:
            st.subheader("Narratives Idenitified")
            grouped = semantic_group_narratives(all_narratives)

            for narrative, count in grouped.items():
                chunks = narrative_chunks.get(narrative, [])
                with st.expander(f"{narrative} ({count} mentions)"):
                    st.write(f"Appears in chunks: {chunks}")
                    st.write("Supporting excerpts and analysis available in per-chunk expanders above.")

            # --- Prepare TXT for download ---
            txt_content = "=== Per-Chunk Analysis ===\n\n"
            for i, res in enumerate(results):
                txt_content += f"--- Chunk {i+1} ---\n{res.strip()}\n\n"

            txt_content += "=== Grouped Similar Narratives ===\n\n"
            for rep, cluster_narratives in grouped.items():
                # Collect chunks for all narratives in this cluster
                cluster_chunks = sorted({chunk for n in cluster_narratives for chunk in narrative_chunks.get(n, [])})
                txt_content += f"{rep} ({len(cluster_narratives)} mentions) — Appears in chunks: {cluster_chunks}\n"

            # Streamlit download button
            st.download_button(
                "Download Analysis (TXT)",
                data=txt_content.encode("utf-8"),
                file_name="disinformation_analysis.txt",
                mime="text/plain"
            )

        else:
            st.info("No distinct narratives identified.")
