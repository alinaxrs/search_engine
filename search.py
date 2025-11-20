import json
from nltk.stem import PorterStemmer
import heapq


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
    stemmed_terms = [stemmer.stem(term) for term in terms]
    for term in stemmed_terms:
        positions.append(term_index[term])

    with open("terms_aggregated.ndjson", "r") as f:
        postings = []
        for position in positions:
            f.seek(position)
            line = f.readline()
            data = json.loads(line)
            postings.append(data["postings"])
    return merge_postings(postings)

def extract_terms(query: str):
    return [i.lower() for i in query.split(" ")]

def main():
    while True:
        query = input("Look for anything: ")
        if query.lower() == "exit":
            break
        terms = extract_terms(query)
        urls = search(terms)
        print(urls[:5])

if __name__ == "__main__":
    main()