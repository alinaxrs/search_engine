import json 
import tokenizer
import zipfile
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
from posting import Posting


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
                
    print(index)
    return index


if __name__ == "__main__":
    index = indexer("developer.zip")
