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

def extract_text(soup):
    try:
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def extract_important_words(soup):
    important = []

    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if text:
            important.append(text)

    for tag in soup.find_all(["strong", "b"]):
        text = tag.get_text(" ", strip=True)
        if text:
            important.append(text)

    if soup.title:
        text = soup.title.get_text(" ", strip=True)
        if text:
            important.append(text)

    return " ".join(important)

def indexer(corpus):
    index = {}
    stemmer = PorterStemmer()
    num_docs = 0
    unique_tokens = 0

    if not os.path.isdir(EXTRACT_PATH):
        with zipfile.ZipFile(corpus, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    count = 1
    for root, _, files in os.walk(EXTRACT_PATH):
        for file in files:
            num_docs += 1

            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            html = data["content"]
            url = data["url"].split("#")[0]

            print("success: ", count, file)

            # text extraction + stemming
            soup = BeautifulSoup(html, "html.parser")
            plain = extract_text(soup)
            tokens = tokenizer.tokenize(plain)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]

            # important words
            important_text = extract_important_words(soup)
            important_tokens = tokenizer.tokenize(important_text)
            important_stemmed = [stemmer.stem(token) for token in important_tokens]

            # 4) build index
            ####### TO DO: create helper function for tokens with importance + freq added
            for word in stemmed_tokens:
                if word not in index:
                    posting = Posting(doc_id=url, freq=1)
                    index[word] = [posting]
                    unique_tokens += 1

                posting = next((p for p in index[word] if p.doc_id == url), None)
                if posting:
                    posting.freq += 1
                else:
                    index[word].append(Posting(doc_id=url, freq=1))
            count += 1
            if count % BATCH_SIZE == 0:
                generate_report(index, num_docs, unique_tokens, f"report_{count}.txt", f"index_{count}.json")
                index = {}

    return index

def generate_report(index, num_docs, unique_tokens, report_file="report.txt", index_file="index_0000.json"):
    # Save index to disk to measure size
    with open(index_file, "w") as f:
        json.dump({k:[p.__dict__ for p in v] for k,v in index.items()}, f, indent=2)
        
    index_size_kb = os.path.getsize(index_file) / 1024

    # Write to report.txt
    with open(report_file, "w") as f:
        f.write(f"Documents indexed: {num_docs}\n")
        f.write(f"Unique tokens: {unique_tokens}\n")
        f.write(f"Index size (KB): {index_size_kb:.2f}\n")

    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    index = indexer("developer.zip")
    generate_report(index)
