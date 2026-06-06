import streamlit as st
import pyterrier as pt
import nltk
from nltk.corpus import wordnet
from sentence_transformers import CrossEncoder
import re
import pandas as pd
from nltk.corpus import stopwords

# Page Configuration
st.set_page_config(page_title="AP News Search Engine", layout="wide")

# --- 1. SYSTEM INITIALIZATION & CACHING ---
@st.cache_resource
def init_system():
    if not pt.java.started():
        pt.java.init()
    
    index = pt.IndexFactory.of("./ap_index")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Load model from local directory
    neural_model = CrossEncoder('./local_minilm', max_length=512)
    
    # Load Topics and Qrels for Ground Truth Evaluation
    topics = pt.io.read_topics("./dataset/topics1-50.txt", format="trec")
    qrels = pt.io.read_qrels("./dataset/qrels1-50ap.txt")
    
    return index, neural_model, topics, qrels

index, neural_model, topics, qrels = init_system()

# --- 2. PIPELINES CONFIGURATION ---
stop_words = set(stopwords.words('english'))

def expand_with_wordnet(row):
    original_query = row["query"]
    new_words = []
    for word in original_query.split():
        if word.lower() in stop_words:
            continue
        synsets = wordnet.synsets(word)
        if synsets:
            found = False
            for synset in synsets[:3]:
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ')
                    if syn.lower() != word.lower() and syn.isalnum():
                        new_words.append(syn)
                        found = True
                        break
                if found:
                    break
            if not found:
                for synset in synsets[:3]:
                    for hypernym in synset.hypernyms():
                        for lemma in hypernym.lemmas():
                            syn = lemma.name().replace('_', ' ')
                            if syn.lower() != word.lower() and syn.isalnum():
                                new_words.append(syn)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
    return original_query + " " + original_query + " " + " ".join(new_words)

class MiniLM_ReRanker(pt.Transformer):
    def transform(self, df):
        df["text"] = df["text"].fillna("") 
        pairs = list(zip(df["query"], df["text"]))
        df["score"] = neural_model.predict(pairs)
        
        # Sort and assign ranks
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index
        return df

bm25 = pt.terrier.Retriever(index, wmodel="BM25")
get_text = pt.text.get_text(index, "text") 

pipelines = {
    "1. BM25 Baseline": bm25 >> get_text,
    "2. BM25 + Bo1 (Statistical PRF)": bm25 >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=10, fb_terms=10) >> bm25 >> get_text,
    "3. BM25 + RM3 (Probabilistic PRF)": bm25 >> pt.rewrite.RM3(index, fb_docs=10, fb_terms=10) >> bm25 >> get_text,
    "4. BM25 + WordNet (Semantic Expansion)": pt.apply.query(expand_with_wordnet) >> bm25 >> get_text,
    "5. BM25 + Neural Re-rank (MiniLM)": (bm25 % 100) >> get_text >> MiniLM_ReRanker()
}

# --- HELPER: CLEAN PYTERRIER QUERY FORMAT ---
def clean_expanded_query(raw_query_str):
    """Strips Terrier formatting (e.g., applypipeline:off and ^0.032 weights) for UI display."""
    cleaned = raw_query_str.replace("applypipeline:off", "")
    cleaned = re.sub(r'\^[0-9.]+', '', cleaned)
    return " ".join(cleaned.split())

# --- 3. UI DESIGN ---
st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj > img { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #4CAF50; }
    
    .article-text {
        font-family: 'Georgia', serif; 
        font-size: 1.15rem; 
        line-height: 1.8; 
        color: #e2e8f0; 
        text-align: justify; 
        padding: 20px;
        background-color: #1e293b; 
        border-radius: 8px;
        border-left: 5px solid #4CAF50; 
        margin-top: 10px;
    }
    
    .badge-relevant { color: #4CAF50; font-weight: bold; }
    .badge-not-relevant { color: #F44336; font-weight: bold; }
    .badge-unjudged { color: #FFC107; font-weight: bold; }
    
    /* Presentation-Ready Expanded Query Box */
    .expanded-query-box {
        background-color: #0f172a; 
        color: #38bdf8; 
        padding: 25px;
        border-radius: 8px;
        border: 2px solid #0284c7;
        font-size: 1.1rem; 
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 25px;
        line-height: 1.6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .expanded-query-title {
        color: #e2e8f0; 
        font-size: 1.2rem;
        font-family: 'Arial', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        display: block;
        font-weight: normal;
    }
    
    /* Breakdown Dashboard Box */
    .breakdown-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #8b5cf6; /* Purple accent */
        margin-top: 15px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Associated Press (1988) Search Engine")
st.markdown("Perform a search to evaluate different Information Retrieval (IR) configurations.")

if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'search_performed' not in st.session_state:
    st.session_state['search_performed'] = False
if 'modified_query' not in st.session_state:
    st.session_state['modified_query'] = ""
if 'qrels_dict' not in st.session_state:
    st.session_state['qrels_dict'] = None
if 'matched_qid' not in st.session_state:
    st.session_state['matched_qid'] = None

with st.sidebar:
    st.header("⚙️ Search Settings")
    selected_model = st.selectbox("Select Retrieval Pipeline:", list(pipelines.keys()))
    display_count = st.slider("Number of Results to Display:", min_value=5, max_value=50, value=10)
    
    st.markdown("---")
    st.markdown("**About the Models:**\n"
                "- **BM25:** Highly effective baseline model relying on exact lexical matching.\n"
                "- **Bo1 / RM3:** Corpus-based PRF methods that enrich queries using top retrieved documents.\n"
                "- **WordNet:** Knowledge-based expansion utilizing lexical synonyms.\n"
                "- **Neural:** Two-stage retrieval where MiniLM captures semantic context to re-rank documents.")
    st.markdown("---")
    st.caption("Yunus Emre Keleş | IR Project")

with st.form(key='search_form'):
    col1, col2 = st.columns([5, 1])
    with col1:
        raw_query = st.text_input("Enter your query:", label_visibility="collapsed", placeholder="Search topics (e.g., space program, oil prices)...")
    with col2:
        submit_button = st.form_submit_button(label='🔍 Search')

if submit_button and raw_query:
    with st.spinner(f'Retrieving results using "{selected_model}"...'):
        try:
            model = pipelines[selected_model]
            results = model.search(raw_query)
            
            st.session_state['results'] = results
            st.session_state['search_performed'] = True
            
            if not results.empty and "query" in results.columns:
                raw_expanded = results["query"].iloc[0]
                st.session_state['modified_query'] = clean_expanded_query(raw_expanded)
            else:
                st.session_state['modified_query'] = raw_query
            
            matched_topic = topics[topics['query'].str.lower() == raw_query.lower()]
            if not matched_topic.empty:
                qid = matched_topic.iloc[0]['qid']
                relevant_docs = qrels[qrels['qid'] == qid].set_index('docno')['label'].to_dict()
                st.session_state['qrels_dict'] = relevant_docs
                st.session_state['matched_qid'] = qid
            else:
                st.session_state['qrels_dict'] = None
                st.session_state['matched_qid'] = None
                
        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")
            st.session_state['search_performed'] = False

if st.session_state['search_performed'] and st.session_state['results'] is not None:
    results = st.session_state['results']
    
    if st.session_state['modified_query'] and st.session_state['modified_query'] != raw_query:
        st.markdown(f"""
            <div class='expanded-query-box'>
                <span class='expanded-query-title'>Expanded / Modified Query:</span>
                {st.session_state['modified_query']}
            </div>
        """, unsafe_allow_html=True)
    
    if st.session_state['matched_qid']:
        st.success(f"🎯 **Dataset Query Matched!** (Topic ID: `{st.session_state['matched_qid']}`). Ground truth relevance evaluation is active.")
        
        # Calculate Top 50 Breakdown
        top_50 = results.head(50)
        rel_count = 0
        not_rel_count = 0
        unjudged_count = 0
        
        q_dict = st.session_state['qrels_dict']
        for doc in top_50['docno']:
            if doc in q_dict:
                if q_dict[doc] > 0:
                    rel_count += 1
                else:
                    not_rel_count += 1
            else:
                unjudged_count += 1
                
        # Display the Breakdown Dashboard
        st.markdown(f"""
            <div class='breakdown-box'>
                <span style='font-size: 1.1rem; color: #e2e8f0; margin-bottom: 10px; display: block;'><strong>Top 50 Documents Breakdown:</strong></span>
                <span style='font-size: 1.25rem;'>
                    <span class='badge-relevant'>✅ {rel_count} Relevant</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    <span class='badge-not-relevant'>❌ {not_rel_count} Not Relevant</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
                    <span class='badge-unjudged'>❔ {unjudged_count} Unjudged</span>
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    if len(results) > 0:
        st.markdown("---")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(label="Total Documents Found", value=f"{len(results)}")
        metric_col2.metric(label="Currently Displaying", value=f"Top {display_count}")
        st.markdown("<br>", unsafe_allow_html=True)
        
        for i, row in results.head(display_count).iterrows():
            with st.container():
                raw_text = str(row['text'])
                docno = row['docno']
                
                relevance_html = ""
                if st.session_state['qrels_dict'] is not None:
                    qrels_dict = st.session_state['qrels_dict']
                    if docno in qrels_dict:
                        label = qrels_dict[docno]
                        if label > 0:
                            relevance_html = " | <span class='badge-relevant'>✅ Relevant</span>"
                        else:
                            relevance_html = " | <span class='badge-not-relevant'>❌ Not Relevant</span>"
                    else:
                        relevance_html = " | <span class='badge-unjudged'>❔ Unjudged</span>"
                
                raw_text = raw_text.replace("``", '"').replace("''", '"').replace("`", "'").replace("$", "&#36;")
                
                paragraphs = re.split(r'\n\s*\n', raw_text)
                clean_paragraphs = [" ".join(p.split()) for p in paragraphs if p.strip()]
                text_html = "<br><br>".join(clean_paragraphs)
                
                words = clean_paragraphs[0].split() if clean_paragraphs else []
                dynamic_title = " ".join(words[:10]) + "..." if len(words) > 10 else " ".join(words)
                
                st.subheader(f"#{i+1} 📰 {dynamic_title}")
                
                st.markdown(f"<span style='font-size: 0.9em; color: gray;'>**Document ID:** <code>{docno}</code> | **Relevance Score:** <code>{row['score']:.4f}</code>{relevance_html}</span>", unsafe_allow_html=True)
                
                with st.expander("Read Full Article", expanded=(i==0)): 
                    st.markdown(f"<div class='article-text'>{text_html}</div>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.warning("No relevant documents found for this query.")