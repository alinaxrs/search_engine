import json
from nltk.stem import PorterStemmer
import math
from collections import defaultdict
import time
import hashlib

# Global variables for index metadata
TOTAL_DOCS = 0
DOC_LENGTHS = {}  # doc_id -> total term count in document
TERM_INDEX = None  # Cache term index in memory
DOC_FINGERPRINTS = {}  # hash -> canonical_doc_id mapping for exact duplicates
DOC_SIMHASHES = {}  # doc_id -> simhash value for near duplicates
SIMHASH_THRESHOLD = 3  # Hamming distance threshold for near-duplicates

def compute_exact_hash(content):
    """Compute MD5 hash for exact duplicate detection"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def compute_simhash(tokens, hash_bits=64):
    """
    Compute SimHash for near-duplicate detection
    tokens: list of stemmed terms from document
    """
    # Initialize vector with zeros
    vector = [0] * hash_bits
    
    for token in tokens:
        # Hash each token
        token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        
        # Update vector based on hash bits
        for i in range(hash_bits):
            if token_hash & (1 << i):
                vector[i] += 1
            else:
                vector[i] -= 1
    
    # Convert vector to fingerprint
    fingerprint = 0
    for i in range(hash_bits):
        if vector[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two integers"""
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance

def remove_exact_duplicates(doc_ids):
    """Remove exact duplicate documents from results"""
    if not DOC_FINGERPRINTS:
        return doc_ids
    
    seen_fingerprints = set()
    unique_docs = []
    
    for doc_id in doc_ids:
        # Get the canonical doc_id for this document
        fingerprint = DOC_FINGERPRINTS.get(doc_id, doc_id)
        
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            unique_docs.append(doc_id)
    
    return unique_docs

def remove_near_duplicates(doc_ids):
    """Remove near-duplicate documents using SimHash"""
    if not DOC_SIMHASHES:
        return doc_ids
    
    unique_docs = []
    seen_hashes = []
    
    for doc_id in doc_ids:
        doc_hash = DOC_SIMHASHES.get(doc_id)
        
        if doc_hash is None:
            # If there is no hash available, keep document
            unique_docs.append(doc_id)
            continue
        
        # Check if this document is similar to any we've seen
        is_duplicate = False
        for seen_hash in seen_hashes:
            if hamming_distance(doc_hash, seen_hash) <= SIMHASH_THRESHOLD:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_docs.append(doc_id)
            seen_hashes.append(doc_hash)
    
    return unique_docs

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

    # Load in duplicate detection data
    try:
        with open("doc_fingerprints.json", "r") as f:
            DOC_FINGERPRINTS = json.load(f)
    except FileNotFoundError:
        print("Warning: doc_fingerprints.json not found")
        DOC_FINGERPRINTS = {}

    try:
        with open("doc_simhashes.json", "r") as f:
            DOC_SIMHASHES = json.load(f)
    except FileNotFoundError:
        print("Warning: doc_simhashes.json not found")
        DOC_SIMHASHES = {}

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



def search(terms: list[str], max_results=100, remove_duplicates=True, duplicate_method='near'):
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
    
    # Boolean AND filtering:
    # Build set of doc_ids for each term
    term_doc_sets = []
    for postings in postings_list:
        if postings:  # Skip empty postings
            doc_set = {posting["doc_id"] for posting in postings}
            term_doc_sets.append(doc_set)
    
    # Find intersection - documents containing ALL terms
    if term_doc_sets:
        valid_docs = set.intersection(*term_doc_sets)
        # Filter ranked results to only include valid docs
        ranked = [doc_id for doc_id in ranked if doc_id in valid_docs]
    
    # Remove duplicates if requested
    if remove_duplicates:
        original_count = len(ranked)
        if duplicate_method == 'exact':
            ranked = remove_exact_duplicates(ranked)
        elif duplicate_method == 'near':
            ranked = remove_near_duplicates(ranked)
        
        # Print duplicate removal stats if any were removed
        removed = original_count - len(ranked)
        if removed > 0:
            print(f"Removed {removed} duplicate(s) using {duplicate_method} detection")
    
    # Return top N results
    return ranked

def extract_terms(query: str):
    """
    Extract alphanumeric terms from query string, lowercased.
    """
    query = query.strip()
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
    print("Enter 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Look for anything: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        terms = extract_terms(query)
        
        start = time.time()
        urls = search(terms, remove_duplicates=True, duplicate_method='near')
        end = time.time()
        
        print(f"Found {len(urls)} {'results' if len(urls) > 1 else 'result'} in {end - start:.4f} seconds")
        if urls:
            print(f"Top {5 if len(urls) >= 5 else len(urls)} {'results' if len(urls) > 1 else 'result'}:")
            for i, url in enumerate(urls[:5], 1):
                print(f"  {i}. {url}")
        else:
            print("No results found.")
        print()

if __name__ == "__main__":
    main()