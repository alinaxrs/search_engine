import json


def merge_postings(postings):
    return []

def search(terms):
    with open("term_index.json", "r") as f:
        term_index = json.load(f)

    positions = []
    for term in terms:
        positions.append(term_index[term])

    with open("terms_aggregated.ndjson", "r") as f:
        postings = []
        for position in positions:
            f.seek(position)
            line = f.readline()
            data = json.loads(line)
            postings.append(data["postings"])
    return merge_postings(postings)

def extract_terms(query):
    return query.split(" AND ")

def main():
    while True:
        query = input("Look for anything: ")
        terms = extract_terms(query)
        urls = search(terms)
        print(urls[:5])