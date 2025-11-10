from dataclasses import dataclass

@dataclass
class Posting:
    doc_id: str
    freq: int