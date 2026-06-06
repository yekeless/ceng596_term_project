import pyterrier as pt
import os
import re
import shutil

# 1. INITIALIZE PYTERRIER
if not pt.java.started():
    pt.java.init()

# --- 2. FILE PATHS ---
corpus_dir = "./dataset/corpus/coll" 
index_path = "./ap_index"

# Remove old index completely to avoid ghost data
if os.path.exists(index_path):
    shutil.rmtree(index_path)
    print("Old index cleared...")

# 3. LIST ALL FILES IN CORPUS DIRECTORY
file_list = []
for file_name in os.listdir(corpus_dir):
    full_path = os.path.join(corpus_dir, file_name)
    if os.path.isfile(full_path):
        file_list.append(full_path)

print(f"Found {len(file_list)} files in [{corpus_dir}] directory.")
print("Starting indexing with text extraction for the neural model...\n")

# --- 4. CUSTOM TREC PARSER FOR NEURAL IR ---
def trec_parser(file_paths):
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Split content into individual document blocks
            documents = content.split('<DOC>')
            for document in documents:
                if not document.strip(): continue
                
                # Extract DOCNO and TEXT tags using Regex
                docno_match = re.search(r'<DOCNO>\s*(.*?)\s*</DOCNO>', document, re.IGNORECASE | re.DOTALL)
                text_matches = re.findall(r'<TEXT>\s*(.*?)\s*</TEXT>', document, re.IGNORECASE | re.DOTALL)
                
                if docno_match and text_matches:
                    text_content = " ".join(text_matches).strip()
                    # Remove any remaining SGML/HTML tags
                    text_content = re.sub(r'<[^>]+>', ' ', text_content)
                    
                    yield {
                        'docno': docno_match.group(1).strip(),
                        'text': text_content
                    }

# --- 5. CONFIGURE ITERDICTINDEXER ---
# Preserve Tokenizer and Stemmer settings
custom_properties = {
    "termpipelines": "Stopwords,PorterStemmer",
    "tokeniser": "EnglishTokeniser"
}

# Allocate 30 chars for docno and 15000 chars for text metadata to prevent truncation warnings
indexer = pt.IterDictIndexer(
    index_path, 
    blocks=True, 
    overwrite=True, 
    meta={'docno': 30, 'text': 15000},
    properties=custom_properties
)

# Pass the custom parser to the indexer
index_ref = indexer.index(trec_parser(file_list))

print("\nIndexing completed successfully. Texts are saved for the neural model.")
print(f"Index location: {index_ref}")