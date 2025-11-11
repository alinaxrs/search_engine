import json 
import tokenizer
import zipfile
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
from posting import Posting
import os


"""
Add functions here to 1) strip html tags to get raw text and 2) extract the texts from 
important tags like h2, strong tags.
"""

EXTRACT_PATH = "developer"
BATCH_SIZE = 2000

def extract_text(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def indexer(corpus):
    """
    Indexer:
    Make a dictionary with key = word, value = html_name
    """
    index = {}
    stemmer = PorterStemmer()

    """
    Open the file with read permission:
        tokenize the file (dont remove stop words)
        add porter stemming to tokens (nltk?)
        concurrently, while going through the html file, regex <h{n}>

    """
    if not os.path.isdir(EXTRACT_PATH):
        with zipfile.ZipFile(corpus, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    count = 1
    for root, _, files in os.walk(EXTRACT_PATH):
        for file in files:
            # 1) open the file
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            html = data["content"]
            url = data["url"].split("#")[0]

            print("success: ", count, file)

            # 2) text extraction + stemming
            plain = extract_text(html)
            tokens = tokenizer.tokenize(plain)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]

            # 3) important words
            important_tokens = []
            # implement imp words logic here
            important_stemmed = [stemmer.stem(token) for token in important_tokens]

            # 4) build index
            for word in stemmed_tokens:
                if word not in index:
                    posting = Posting(doc_id=url, freq=1)
                    index[word] = [posting]

                posting = next((p for p in index[word] if p.doc_id == url), None) # search for a url in the list of postings for a word
                if posting:
                    posting.freq += 1
                else:
                    index[word].append(Posting(doc_id=url, freq=1))
            count += 1
            if count % BATCH_SIZE == 0:
                generate_report(index, f"report_{count}.txt", f"index_{count}.json")
                index = {}

    return index

def generate_report(index, report_file="report.txt", index_file="index.json"):
    # Num of unique documents
    doc_set = set()
    for postings in index.values():
        for p in postings:
            doc_set.add(p.doc_id)
    num_docs = len(doc_set)

    # Num of unique tokens
    num_tokens = len(index)

    # Save index to disk to measure size
    with open(index_file, "w") as f:
        json.dump({k:[p.__dict__ for p in v] for k,v in index.items()}, f, indent=2)
    index_size_kb = os.path.getsize(index_file) / 1024

    # Write to report.txt
    with open(report_file, "w") as f:
        f.write(f"Documents indexed: {num_docs}\n")
        f.write(f"Unique tokens: {num_tokens}\n")
        f.write(f"Index size (KB): {index_size_kb:.2f}\n")

    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    index = indexer("developer.zip")
    generate_report(index)
