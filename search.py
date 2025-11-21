import json
from nltk.stem import PorterStemmer
# import heapq
import time

def merge_postings(postings):
    postings = sorted(postings, key=len)
    common = {p["doc_id"] for p in postings[0]}
    for posting in postings[1:]:
        common &= {p["doc_id"] for p in posting}
    return [c for c in common]



def search(terms: list[str]):
    stemmer = PorterStemmer()
    
    with open("term_index.json", "r") as f:
        term_index = json.load(f)

    positions = []
    for term in terms:
        stemmed = stemmer.stem(term)
        positions.append(term_index[stemmed])

    with open("index.ndjson", "r") as f:
        postings = []
        for position in positions:
            f.seek(position)
            line = f.readline()
            data = json.loads(line)
            postings.append(data["postings"])
    return merge_postings(postings)

def extract_terms(query: str):
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
    return result

def main():
    while True:
        query = input("Look for anything: ")
        if query.lower() == "exit":
            break
        terms = extract_terms(query)
        start = time.time()
        urls = search(terms)
        end = time.time()
        print(urls[:5])
        print(f"\nSearch took {end - start:.4f} seconds\n")

if __name__ == "__main__":
    main()