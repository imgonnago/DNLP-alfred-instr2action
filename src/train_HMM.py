import json

from src.make_vocab import load_data
from src.models.HMM import HMM
from src.evaluate import evaluate
from src.utils import load_config, resolve

def main():
    cfg = load_config()
    out_dir = resolve("outputs")
    out_dir.mkdir(parents = True, exist_ok = True)

    model = HMM()
    train = [model.prepare(l) for l in load_data("train")]
    valid = [model.prepare(l) for l in load_data("valid")]
    test = [model.prepare(l) for l in load_data("test")]
    unk = model.words_stoi["<unk>"]

    model.count(train)
    assert model.pi_count.sum() == len(train)

    results = {}
    for k in cfg["hmm"]["k_candidates"]:
        model.k = k
        model.smooth()
        metrics, _ = evaluate(model.viterbi, valid, unk)
        results[k] = metrics
        print(f"k={k:<6} valid {metrics}")

    best_k = max(results, key=lambda k: results[k]["step_acc"])
    model.k = best_k
    model.smooth()
    model.save(out_dir / "hmm.npz")

    test_metrics, test_preds = evaluate(model.viterbi, test, unk)
    print(f"best k={best_k}  test {test_metrics}")

    n_special = len(cfg["tokens"]["actions_special"])
    assert all(a >= n_special for p in test_preds for a in p)

    with open(out_dir / "hmm_results.json", "w", encoding="utf-8") as f:
        json.dump({"valid": {str(k): v for k, v in results.items()},
                   "best_k": best_k, "test": test_metrics}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()