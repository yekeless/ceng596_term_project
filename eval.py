import pyterrier as pt
import nltk
from nltk.corpus import wordnet
from sentence_transformers import CrossEncoder
import os

# Load NLTK dictionaries
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

if not pt.java.started():
    pt.java.init()

# 1. FILE PATHS & DATA LOADING
index_path = "./ap_index"     
topics_file = "dataset/topics1-50.txt" 
qrels_file = "dataset/qrels1-50ap.txt" 

print("Loading data and index...")
topics = pt.io.read_topics(topics_file, format="trec")
qrels = pt.io.read_qrels(qrels_file)
index = pt.IndexFactory.of(index_path)

# 2. WORDNET EXPANSION FUNCTION
from nltk.corpus import stopwords
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

# 3. NEURAL RE-RANKER
model_dir = './local_minilm'
if not os.path.exists(model_dir):
    print("Downloading model...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    model.save(model_dir)
else:
    print("Loading model from local directory (offline)...")
    model = CrossEncoder(model_dir, max_length=512)

class MiniLM_ReRanker(pt.Transformer):
    def transform(self, df):
        df["text"] = df["text"].fillna("")
        pairs = list(zip(df["query"], df["text"]))
        df["score"] = model.predict(pairs)
        return df

# 4. BASE RETRIEVER
bm25 = pt.terrier.Retriever(index, wmodel="BM25")

fb_docs_values = [5, 10, 20, 30, 50, 100]
fb_terms_values = [5, 10, 20, 30, 50, 100]

# --- BO1 PARAMETER SEARCH ---
print("\n--- SEARCHING FOR OPTIMAL BO1 PARAMETERS (36 combinations) ---")

bo1_pipelines = []
bo1_names = []

for fb_docs in fb_docs_values:
    for fb_terms in fb_terms_values:
        pipeline = bm25 >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> bm25
        bo1_pipelines.append(pipeline)
        bo1_names.append(f"Bo1 docs={fb_docs} terms={fb_terms}")

bo1_results = pt.Experiment(
    bo1_pipelines,
    topics,
    qrels,
    eval_metrics=["map", "P_10", "ndcg"],
    names=bo1_names
)

print(bo1_results.to_string())

best_bo1_row = bo1_results.loc[bo1_results["map"].idxmax()]
best_bo1_name = best_bo1_row["name"]
best_bo1_docs = int(best_bo1_name.split("docs=")[1].split(" ")[0])
best_bo1_terms = int(best_bo1_name.split("terms=")[1])
print(f"\nBest Bo1: fb_docs={best_bo1_docs}, fb_terms={best_bo1_terms} (MAP={best_bo1_row['map']:.4f})")

# --- RM3 PARAMETER SEARCH ---
print("\n--- SEARCHING FOR OPTIMAL RM3 PARAMETERS (144 combinations) ---")

fb_lambda_values = [0.3, 0.5, 0.6, 0.8]

rm3_pipelines = []
rm3_names = []

for fb_docs in fb_docs_values:
    for fb_terms in fb_terms_values:
        for fb_lambda in fb_lambda_values:
            pipeline = bm25 >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=fb_lambda) >> bm25
            rm3_pipelines.append(pipeline)
            rm3_names.append(f"RM3 docs={fb_docs} terms={fb_terms} lambda={fb_lambda}")

rm3_results = pt.Experiment(
    rm3_pipelines,
    topics,
    qrels,
    eval_metrics=["map", "P_10", "ndcg"],
    names=rm3_names
)

print(rm3_results.to_string())

best_rm3_row = rm3_results.loc[rm3_results["map"].idxmax()]
best_rm3_name = best_rm3_row["name"]
best_rm3_docs = int(best_rm3_name.split("docs=")[1].split(" ")[0])
best_rm3_terms = int(best_rm3_name.split("terms=")[1].split(" ")[0])
best_rm3_lambda = float(best_rm3_name.split("lambda=")[1])
print(f"\nBest RM3: fb_docs={best_rm3_docs}, fb_terms={best_rm3_terms}, fb_lambda={best_rm3_lambda} (MAP={best_rm3_row['map']:.4f})")

# --- 5. FINAL EXPERIMENT ---
print("\n--- FINAL 5-MODEL COMPARISON WITH OPTIMAL PARAMETERS ---")

bm25_bo1 = bm25 >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=best_bo1_docs, fb_terms=best_bo1_terms) >> bm25
bm25_rm3 = bm25 >> pt.rewrite.RM3(index, fb_docs=best_rm3_docs, fb_terms=best_rm3_terms, fb_lambda=best_rm3_lambda) >> bm25
bm25_wordnet = pt.apply.query(expand_with_wordnet) >> bm25
get_text = pt.text.get_text(index, "text")
bm25_neural = (bm25 % 100) >> get_text >> MiniLM_ReRanker()

experiment_results = pt.Experiment(
    [bm25, bm25_bo1, bm25_rm3, bm25_wordnet, bm25_neural],
    topics,
    qrels,
    eval_metrics=["map", "ndcg", "P_10"],
    names=[
        "1. BM25 Baseline",
        f"2. BM25 + Bo1 (docs={best_bo1_docs}, terms={best_bo1_terms})",
        f"3. BM25 + RM3 (docs={best_rm3_docs}, terms={best_rm3_terms}, lambda={best_rm3_lambda})",
        "4. BM25 + WordNet",
        "5. BM25 + Neural Re-rank"
    ]
)

print("--- RESULTS ---")
print(experiment_results)