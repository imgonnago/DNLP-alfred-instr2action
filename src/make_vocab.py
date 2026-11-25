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

def load_vocab(vocab_type=None):
    cfg = load_config()
    path = resolve(cfg["paths"]["vocab_dir"]) / f"vocab_{vocab_type}.json"
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    itos = obj["itos"]
    return itos, build_stoi(itos)

def prepend_special_tokens(vocab, vocab_type ="words"):
    cfg = load_config()
    word_speccial = cfg["tokens"]["words_special"]
    action_special = cfg["tokens"]["actions_special"]

    if vocab_type == "words":
        itos = word_speccial + vocab
        print("\n")
        print("[added word special token!]")
    elif vocab_type == "actions":
        itos = action_special + vocab
        print("\n")
        print("[added action special token!]")
    return itos

def build_stoi(vocab):
    stoi = {s : i for i, s in enumerate(vocab)}
    return stoi

def build_itos(stoi):
    itos = {i : s for i, s in enumerate(stoi)}
    return itos

def encode(tokens, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokens]

def decode(ids, itos):
    return [itos[i] for i in ids]

def save_vocab(vocab_type="words", min_freq = None, specials=None, itos=None):
    cfg = load_config()
    VOCAB_DIR = resolve(cfg["paths"]["vocab_dir"])
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    obj = {"min_freq": min_freq, "specials": specials, "size": len(itos), "itos": itos}
    
    if vocab_type == "words":
        with open( VOCAB_DIR / "vocab_words.json", "w",  encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"words vocab 생성 완료: {VOCAB_DIR / 'vocab_words.json'}")
    elif vocab_type == "actions":
        with open( VOCAB_DIR / "vocab_actions.json", "w",  encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"actoins vocab 생성 완료: {VOCAB_DIR / 'vocab_actions.json'}")


def main() -> None:
    cfg = load_config()
    VOCAB_DIR = resolve(cfg["paths"]["vocab_dir"])
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    words_specials = cfg["tokens"]["words_special"]
    words_min_freq = cfg["tokens"]["words_min_freq"]
    action_specials = cfg["tokens"]["actions_special"]
    action_min_freq = cfg["tokens"]["actions_min_freq"]
    data = load_data()
    words_counter = Counter()
    action_counter = Counter()

    # buiild word vocab
    for line in data:
        for sent in line["steps"]:
            words_counter.update(tokenize(sent))

    words = [w for w, cont in words_counter.items() if cont >= words_min_freq]
    words = sorted(words, key=lambda w: (-words_counter[w], w))
    words_itos = prepend_special_tokens(words, vocab_type="words")
    stoi = build_stoi(words_itos)
    save_vocab("words", words_min_freq, words_specials, words_itos)

    print("\n")
    print("[word vocab head 10]")
    for w in words[:10]:
        print(w, words_counter[w])
    print("\n")
    print("[word vocab tail 10]")
    for w in words[-10:]:
        print(w, words_counter[w])

    # build action vocab
    for line in data:
        action_counter.update(line["actions"])

    actions = [a for a, cont in action_counter.items() if cont >= action_min_freq]
    actions = sorted(actions, key=lambda a: (-action_counter[a], a))
    action_itos = prepend_special_tokens(actions, vocab_type="actions")
    save_vocab("actions", action_min_freq, action_specials, action_itos)

    print("\n")
    print("[action vocab head 10]")
    for a in actions[:10]:
        print(a, action_counter[a])
    print("\n")
    print("[action vocab tail 10]")
    for a in actions[-10:]:
        print(a, action_counter[a])

if __name__ == "__main__":
    main()
