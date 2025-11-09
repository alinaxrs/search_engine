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

def extract_text(html):
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)

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
    with zipfile.ZipFile(corpus, "r") as zip_ref:
        for file in zip_ref.namelist():
            try:
                # 1) load json
                with zip_ref.open(file) as f:
                    data = json.load(f)
                print("success: ", file)
                html = data["content"]
                encoding = data["encoding"]
                url = data["url"].split("#")[0]

                # 2) text extraction + stemming
                plain = extract_text(html)
                tokens = tokenizer.tokenize(plain)
                stemmed_tokens = [stemmer.stem(token) for token in tokens]

                # 3) important words
                important_tokens = []
                # implement imp words logic here
                important_stemmed = [stemmer.stem(token) for token in important_tokens]

                # 4) build index
                for i, word in enumerate(stemmed_tokens):
                    if word not in index:
                        posting = Posting(url, 1)
                        index[word] = [posting]
                    if url not in index[word]:
                        posting = Posting(url, 1)
                        index[word].append(posting)
                    else: # url is already associated with the word, so increment the posting's frequency
                        j = index[word].index(url)
                        index[word][j].freq += 1
                        # insert to index, could potentially include tuple with url weight for
                        # when they are different importance levels

            except json.decoder.JSONDecodeError as e:
                print(e)
                print("error:", file)
    return index

def generate_report(index, report_file="report.txt"):
    # Num of unique documents
    doc_set = set()
    for postings in index.values():
        for p in postings:
            doc_set.add(p.url)
    num_docs = len(doc_set)

    # Num of unique tokens
    num_tokens = len(index)

    # Save index to disk to measure size
    with open("index.json", "w") as f:
        json.dump({k:[p.__dict__ for p in v] for k,v in index.items()}, f)
    index_size_kb = os.path.getsize("index.json") / 1024

    # Write to report.txt
    with open(report_file, "w") as f:
        f.write(f"Documents indexed: {num_docs}\n")
        f.write(f"Unique tokens: {num_tokens}\n")
        f.write(f"Index size (KB): {index_size_kb:.2f}\n")

    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    index = indexer("developer.zip")
    generate_report(index)
