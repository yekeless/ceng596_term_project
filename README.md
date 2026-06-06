# 📰 Associated Press (1988) Search Engine & IR Evaluation Dashboard

**Author:** Yunus Emre Keleş | Baran Boloğur
**Course:** CENG 596 

## 📝 Abstract
This project investigates the impact of query expansion and neural re-ranking on information retrieval performance using the AP88 dataset. We built a multi-stage retrieval pipeline in PyTerrier, evaluating a BM25 baseline against three query expansion techniques: statistical pseudo-relevance feedback (Bo1 and RM3) and knowledge-based semantic expansion (WordNet). Furthermore, we integrated a two-stage dense neural architecture using a MiniLM Cross-Encoder. To facilitate qualitative analysis, we developed a reactive Streamlit dashboard that dynamically maps retrieved documents to ground-truth relevance assessments. 

## ✨ Key Features
* **Multiple Retrieval Pipelines:** Compare BM25 strictly against Bo1, RM3, WordNet, and Neural Re-ranking.
* **"Under the Hood" Visibility:** UI displays the exact mathematically expanded queries.
* **Ground-Truth Evaluation:** Automatically maps retrieved documents to TREC Qrels, displaying "Relevant", "Not Relevant", and "Unjudged" badges.
* **Top 50 Breakdown:** Live calculation of precision metrics on the UI.
