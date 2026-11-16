from collections import Counter
import re
import json
from utils import resolve, load_config

def tokenize(text):
    return re.findall(r"[a-z]+|\d+", text.lower())

def load_data(split="train"):
    cfg = load_config()
    path = resolve(cfg["paths"]["processed_dir"]) / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    print(split, len(samples))
    return samples

def main():
    data = load_data()
    counter = Counter()
    for line in data:
        for sent in line["steps"]:
            counter.update(tokenize(sent))
    min_freq = 3
    words = [w for w, cont in counter.items() if cont >= min_freq]
    words = sorted(words, key=lambda w: (-counter[w], w))

    for w in words[:10]:
        print(w, counter[w])
    for w in words[-10:]:
        print(w, counter[w])


if __name__ == "__main__":
    main()
