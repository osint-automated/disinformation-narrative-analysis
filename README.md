# Disinformation Narrative Analysis

This tool automates the detection and clustering of disinformation and influence narratives within text-based articles. It implements an applied workflow for automated narrative intelligence, combining LLM-based reasoning with semantic similarity modeling to surface, categorize, and contextualize disinformation patterns across large text samples.

A live version of this tool is available here: [https://disinformation-narrative-analysis.streamlit.app/](https://disinformation-narrative-analysis.streamlit.app/)

A free version requiring no API key is available here: [https://lashunda-unsultry-frilly.ngrok-free.dev/](https://lashunda-unsultry-frilly.ngrok-free.dev/)

## How It Works

The application follows a multi-step process to analyze text:

1.  **Input Handling**: Users can upload a `.txt` file or paste text manually. The application segments the content into manageable chunks for large-scale text processing.
2.  **LLM-Powered Analysis**: Each text chunk is processed through the OpenAI API (`gpt-4o-mini`). The model identifies disinformation narratives, extracts supporting excerpts, and provides concise analytical context.
3.  **Semantic Grouping**: Using `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) and `AgglomerativeClustering`, the system groups thematically similar narratives based on cosine similarity, consolidating overlapping insights.
4.  **Interactive Results**: A Streamlit interface visualizes the per-chunk analyses, clusters related narratives, and provides a downloadable `.txt` report summarizing findings and cluster associations.

## Usage

1.  **Enter your OpenAI API Key**: Provide your OpenAI API key to enable the LLM analysis. Your key is not stored.
2.  **Choose Input Method**:
    *   **Upload a .txt file**: Upload a text file containing the article you wish to analyze.
    *   **Paste text manually**: Paste the article content directly into the text area.
3.  **Run Analysis**: Click the "Run Analysis" button to start the process.
4.  **Review Results**:
    *   **Per-Chunk Analysis**: Expand each chunk to see the detailed LLM analysis.
    *   **Narratives Identified**: View a consolidated list of unique narratives, their mention count, and which chunks they appear in.
    *   **Download Analysis (CSV)**: Download a CSV file summarizing the identified narratives.

## Intent

The primary intent of this application is to assist researchers, analysts, and the general public in understanding the underlying narratives present in various texts, especially those that might contain disinformation. By highlighting these narratives and providing supporting excerpts and analysis, the tool aims to foster critical thinking and informed decision-making.