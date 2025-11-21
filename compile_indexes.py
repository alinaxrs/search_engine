import os
import json
from collections import defaultdict
import heapq

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def convert_index_json_to_sorted_ndjson(index_json_path, out_ndjson_path):
    """
    Convert's single index JSON file into a sorted NDJSON file with one object per
    line: {"term":..., "postings": [...]}, sorted by term alphabetically.
    """
    print(f"Converting {index_json_path} -> {out_ndjson_path}")
    with open(index_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = sorted(data.items(), key=lambda x: x[0])
    with open(out_ndjson_path, "w", encoding="utf-8") as out:
        for term, postings in items:
            out.write(json.dumps({"term": term, "postings": postings}, ensure_ascii=False) + "\n")


def k_way_merge_partials_to_terms(partial_files, terms_file):
    """
    K-way merge to sort and aggregate postings from all partial NDJSON files.
    Each partial line is {"term":..., "postings": [...]} as shown in convert_index_json_to_sorted_ndjson.
    This function produces `terms_file` where each
    line is {"term":..., "sf": <int>, "postings": [...] } with merged postings.
    """
    print(f"K-way merging {len(partial_files)} partials -> {terms_file}")

    # Open all partials and read first line
    files = [open(p, "r", encoding="utf-8") for p in partial_files]
    try:
        heap = []  # entries: (term, file_index, postings)
        for idx, f in enumerate(files):
            line = f.readline()
            if not line:
                continue
            obj = json.loads(line)
            heap.append((obj["term"], idx, obj["postings"]))

        heapq.heapify(heap)

        term_index = {}

        with open(terms_file, "w", encoding="utf-8") as out:
            while heap:
                term, idx, postings = heapq.heappop(heap)
                merged_docs = defaultdict(int)

                term_index[term] = out.tell()

                # incorporate postings from the popped entry
                for p in postings:
                    merged_docs[p["doc_id"]] += p["freq"]

                # pull any other entries from heap with the same term
                while heap and heap[0][0] == term:
                    _, idx2, postings2 = heapq.heappop(heap)
                    for p in postings2:
                        merged_docs[p["doc_id"]] += p["freq"]
                    # refill file idx2
                    line2 = files[idx2].readline()
                    if line2:
                        obj2 = json.loads(line2)
                        heapq.heappush(heap, (obj2["term"], idx2, obj2["postings"]))

                # refill file idx for the first popped entry
                line = files[idx].readline()
                if line:
                    obj = json.loads(line)
                    heapq.heappush(heap, (obj["term"], idx, obj["postings"]))

                # write merged term
                postings_list = sorted([{"doc_id": d, "freq": f} for d, f in merged_docs.items()], key=lambda x: x["doc_id"])
                sf = sum(p["freq"] for p in postings_list)
                out.write(json.dumps({"term": term, "sf": sf, "postings": postings_list}, ensure_ascii=False) + "\n")

    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        
        with open("term_index.json", "w") as out:
            json.dump(term_index, out)


def main():
    index_files = []
    starting = "2000"
    for i in range(27):
        index_files.append(f"index_{starting}.json")
        starting = str(int(starting) + 2000)

    # Lecture-style k-way merge flow (convert inputs -> sorted partials -> k-way merge)
    partials_dir = "partials"
    ensure_dir(partials_dir)
    partial_files = []
    for i, fname in enumerate(index_files):
        out = os.path.join(partials_dir, f"partial_{i:02d}.ndjson")
        convert_index_json_to_sorted_ndjson(fname, out)
        partial_files.append(out)

    # K-way merge partials -> index.ndjson
    terms_file = "index.ndjson"
    k_way_merge_partials_to_terms(partial_files, terms_file)

    print(f"K-way merge complete. Aggregated terms written to {terms_file}")


if __name__ == "__main__":
    main()
