from collections import namedtuple

Posting = namedtuple("Posting", ["doc_id", "freq"]) # in M2, change freq to tf-idf score