This tool (available here: https://disinformation-narrative-analysis.streamlit.app/) automates the detection and clustering of disinformation and influence narratives within text-based articles. The tool implements an applied workflow for automated narrative intelligence, combining LLM-based reasoning with semantic similarity modeling to surface, categorize, and contextualize disinformation patterns across large text samples.

Input Handling: Users can upload a .txt file or paste text manually. The application segments the content into manageable chunks for large-scale text processing.

LLM-Powered Analysis: Each text chunk is processed through the OpenAI API (gpt-4o-mini). The model identifies disinformation narratives, extracts supporting excerpts, and provides concise analytical context.

Semantic Grouping: Using SentenceTransformer embeddings (all-MiniLM-L6-v2) and Agglomerative Clustering, the system groups thematically similar narratives based on cosine similarity, consolidating overlapping insights.

Interactive Results: Streamlit visualizes the per-chunk analyses, clusters related narratives, and provides a downloadable .txt report summarizing findings and cluster associations.
