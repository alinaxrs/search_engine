import re

def tokenize(content):
    content = content.lower()
    tokens = re.findall(r"\b[^\W_]+\b", content) 
    return [token for token in tokens]

def computeWordFrequencies(tokens):
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq