import json 
import tokenizer
import re
from nltk.stem import PorterStemmer


"""
Add functions here to 1) strip html tags to get raw text and 2) extract the texts from 
important tags like h2, strong tags.
"""

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
    for file in corpus:

        # 1) load json
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        html = data["content"]
        url = data["url"].split("#")[0]

        # 2) text extraction + stemming
        plain = ""
        # plain = strip_html(html)
        tokens = tokenizer.tokenize(plain)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        # 3) important words
        important_tokens = []
        # implement imp words logic here
        important_stemmed = [stemmer.stem(token) for token in important_tokens]

        # 4) build index
        for i, word in enumerate(stemmed_tokens):
            if word not in index:
                index[word] = {}
            if url not in index[word]:
                pass
                # insert to index, could potentially include tuple with url weight for
                # when they are different importance levels
    return index

if __name__ == "__main__":
    index = indexer(developer)
