# 📰 Associated Press (1988) Search Engine & IR Evaluation Dashboard

**Author:** Yunus Emre Keleş | Baran Boloğur  
**Course:** CENG 596 

## 📝 Abstract
This project investigates the impact of query expansion and neural re-ranking on information retrieval performance using the AP88 dataset. We built a multi-stage retrieval pipeline in PyTerrier, evaluating a BM25 baseline against three query expansion techniques: statistical pseudo-relevance feedback (Bo1 and RM3) and knowledge-based semantic expansion (WordNet). Furthermore, we integrated a two-stage dense neural architecture using a MiniLM Cross-Encoder. To facilitate qualitative analysis, we developed a reactive Streamlit dashboard that dynamically maps retrieved documents to ground-truth relevance assessments. 

## ✨ Key Features
* **Multiple Retrieval Pipelines:** Compare BM25 strictly against Bo1, RM3, WordNet, and Neural Re-ranking.
* **"Under the Hood" Visibility:** UI displays the exact mathematically expanded queries.
* **Ground-Truth Evaluation:** Automatically maps retrieved documents to TREC Qrels, displaying "Relevant", "Not Relevant", and "Unjudged" badges.
* **Top 50 Breakdown:** Live calculation of precision metrics directly on the UI.

## 📥 Prerequisites & Data Download

Due to GitHub's file size constraints, the pre-built PyTerrier index, the AP88 dataset, and the local Neural Model weights are hosted externally.

1. **Download the necessary data files:** 👉 **[Click here to download the required files from Google Drive](https://drive.google.com/drive/folders/1X48kQJqjipz9VjRWiF_-cl4ovmn2mkwF?usp=sharing)**

2. **Extract and Place:** Extract the downloaded files and place the respective folders (e.g., `ap_index`, `dataset`, `local_minilm`) directly into the root directory of this repository.
