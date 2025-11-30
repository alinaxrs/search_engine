import json
from nltk.stem import PorterStemmer
import math
from collections import defaultdict
import time

# Global variables for index metadata
TOTAL_DOCS = 0
DOC_LENGTHS = {}  # doc_id -> total term count in document
TERM_INDEX = None  # Cache term index in memory

# Stopwords to filter out
# STOPWORDS = {
#     'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
#     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
#     'to', 'was', 'will', 'with'
# }

def load_metadata():
    """Load document count and lengths for TF-IDF calculations"""
    global TOTAL_DOCS, DOC_LENGTHS, TERM_INDEX
    
    # Load term index once into memory
    TOTAL_DOCS = 55393
    DOC_LENGTHS = {}
    try:
        with open("term_index.json", "r") as f:
            TERM_INDEX = json.load(f)
    except FileNotFoundError:
        print("Warning: term_index.json not found")
        TERM_INDEX = {}

def compute_tf_idf_scores(query_terms, postings_list, term_dfs):
    """
    Compute TF-IDF scores for documents matching query terms.
    Returns: dict of doc_id -> score, sorted by score descending
    """
    doc_scores = defaultdict(float)
    query_term_weights = defaultdict(int)
    
    # Count query term frequencies
    for term in query_terms:
        query_term_weights[term] += 1
    
    # Compute IDF for each query term
    term_idfs = {}
    for term, df in term_dfs.items():
        if df > 0:
            term_idfs[term] = math.log(TOTAL_DOCS / df)
        else:
            term_idfs[term] = 0
    
    # Calculate document scores
    for term, postings in zip(query_terms, postings_list):
        if not postings:
            continue
            
        idf = term_idfs.get(term, 0)
        
        for posting in postings:
            doc_id = posting["doc_id"]
            tf = posting["freq"]
            
            # TF-IDF with log normalization
            tf_score = 1 + math.log(tf) if tf > 0 else 0
            
            # Document length normalization
            doc_len = DOC_LENGTHS.get(doc_id, 100)  # default avg length
            length_norm = 1 / math.sqrt(doc_len) if doc_len > 0 else 1
            
            # Final score: TF * IDF * length_normalization
            score = tf_score * idf * length_norm
            doc_scores[doc_id] += score
    
    return doc_scores

def merge_postings_ranked(postings_list, query_terms, term_dfs):
    """
    Merge postings with TF-IDF ranking instead of boolean AND.
    Returns list of doc_ids ranked by relevance.
    """
    if not postings_list or all(not p for p in postings_list):
        return []
    
    # Compute TF-IDF scores
    doc_scores = compute_tf_idf_scores(query_terms, postings_list, term_dfs)
    
    if not doc_scores:
        return []
    
    # Sort by score descending
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the doc_ids
    return [doc_id for doc_id, score in ranked_docs]



def search(terms: list[str], max_results=100):
    """Enhanced search with TF-IDF ranking, stopword filtering, and optimization"""
    stemmer = PorterStemmer()
    
    # Use cached term index
    if TERM_INDEX is None:
        print("Error: term_index not loaded.")
        return []
    
    # Filter stopwords and short terms, then stem
    filtered_terms = []
    stemmed_terms = []
    positions = []
    
    for term in terms:
        term_lower = term.lower()
        
        # Skip stopwords and very short terms
        # if term_lower in STOPWORDS or len(term_lower) < 2:
        #     continue
        
        filtered_terms.append(term_lower)
        stemmed = stemmer.stem(term_lower)
        stemmed_terms.append(stemmed)
        
        # Handle missing terms
        if stemmed in TERM_INDEX:
            positions.append(TERM_INDEX[stemmed])
        else:
            positions.append(None)  # Term not in index
    
    # If all terms filtered out, return empty
    if not stemmed_terms:
        return []
    
    # Fetch postings from index file
    postings_list = []
    term_dfs = {}  # document frequencies for IDF calculation
    
    try:
        with open("index.ndjson", "r") as f:
            for i, position in enumerate(positions):
                if position is None:
                    # Term not found in index
                    postings_list.append([])
                    term_dfs[stemmed_terms[i]] = 0
                    continue
                
                try:
                    f.seek(position)
                    line = f.readline()
                    
                    if not line.strip():
                        # Empty line, term not found
                        postings_list.append([])
                        term_dfs[stemmed_terms[i]] = 0
                        continue
                    
                    data = json.loads(line)
                    postings = data.get("postings", [])
                    postings_list.append(postings)
                    
                    # Store document frequency (number of docs containing term)
                    term_dfs[stemmed_terms[i]] = len(postings)
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Handle malformed data
                    postings_list.append([])
                    term_dfs[stemmed_terms[i]] = 0
                    
    except FileNotFoundError:
        print("Error: index.ndjson not found. Run compile_indexes.py first.")
        return []
    
    # Use TF-IDF ranking with early termination
    ranked = merge_postings_ranked(postings_list, stemmed_terms, term_dfs)
    
    # Return top N results for performance
    print(len(ranked), "results found.")
    return ranked[:max_results]

def extract_terms(query: str):
    query = query.strip()
    # print("this is the query: ", query)
    result = []
    word = ""
    for char in query:
        if char.isalnum():
            word += char.lower()
        else:
            if word:
                result.append(word)
                word = ""
    if word:
        result.append(word)
    return result

def main():
    # Load metadata once at startup
    load_metadata()
    # print(f"Search engine ready. Indexed {TOTAL_DOCS} documents.")
    print("Enter 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Look for anything: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        terms = extract_terms(query)
        
        start = time.time()
        urls = search(terms)
        end = time.time()
        
        print(f"Found {len(urls)} results in {end - start:.4f} seconds")
        if urls:
            print("Top 5 results:")
            for i, url in enumerate(urls[:5], 1):
                print(f"  {i}. {url}")
        else:
            print("No results found.")
        print()

if __name__ == "__main__":
    main()