import re

def tokenize(content):
    content = content.lower()
    tokens = re.findall(r'\b\w+\b', content) 
    return [token for token in tokens if token.isalpha() and len(token) > 2]

def computeWordFrequencies(tokens):
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq