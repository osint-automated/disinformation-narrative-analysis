import re
import textwrap
import streamlit as st
import pandas as pd
from openai import OpenAI
from collections import defaultdict
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
    article_text = st.text_area("Paste your article text here:", height=300)

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

# --- Emoji Removal ---
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

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
        chunk_result = remove_emojis(response.choices[0].message.content.replace("**", ""))
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
    merged = {}
    for label, group in clustered.items():
        merged_label = group[0]
        merged[merged_label] = len(group)
    return merged

# --- Run Analysis ---
if st.button("Run Analysis"):
    if not article_text.strip():
        st.warning("Please upload a file or paste text before running analysis.")
        st.stop()

    with st.spinner("Running analysis..."):
        results = analyze_disinformation_llm(article_text)
        st.success("Analysis complete!")

        # Display per-chunk results
        st.subheader("Per-Chunk Analysis")
        for i, res in enumerate(results):
            with st.expander(f"Chunk {i+1}", expanded=False):
                wrapped_text = textwrap.fill(res.strip(), width=120)
                st.markdown(f"<pre style='white-space: pre-wrap;'>{wrapped_text}</pre>", unsafe_allow_html=True)

        # Extract narratives
        narrative_chunks = defaultdict(list)
        all_narratives = []
        for idx, r in enumerate(results):
            matches = re.findall(r"Disinformation Narrative Identified:\s*(.+)", r)
            matches = [m.strip() for m in matches if len(m.strip()) > 2]
            for m in matches:
                narrative_chunks[m].append(idx+1)
            all_narratives.extend(matches)

        if all_narratives:
            st.subheader("Grouped Similar Narratives")
            grouped = semantic_group_narratives(all_narratives)

            # Display grouped narratives
            grouped_text = []
            for narrative, count in grouped.items():
                chunks = narrative_chunks.get(narrative, [])
                st.write(f"{narrative} ({count} mentions) — Appears in chunks: {chunks}")
                grouped_text.append(f"{narrative} ({count} mentions) — Chunks: {chunks}")

            # Prepare TXT for download
            txt_content = "=== Per-Chunk Analysis ===\n\n"
            for i, res in enumerate(results):
                txt_content += f"--- Chunk {i+1} ---\n{res.strip()}\n\n"

            txt_content += "\n=== Grouped Narratives ===\n\n"
            txt_content += "\n".join(grouped_text)

            st.download_button(
                "Download Analysis (TXT)",
                data=txt_content.encode("utf-8"),
                file_name="disinformation_analysis.txt",
                mime="text/plain"
            )
        else:
            st.info("No distinct narratives identified.")
