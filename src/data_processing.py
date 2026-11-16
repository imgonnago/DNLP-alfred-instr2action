import json
from collections import Counter

from src.utils import load_config, resolve

cfg = load_config()
RAW_DIR = resolve(cfg["paths"]["raw_dir"])
OUT_DIR = resolve(cfg["paths"]["processed_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

report = {}
for role, official in cfg["splits"].items():
    stats = Counter(empty=0, misaligned=0, duplicate=0)
    samples, seen = [], set()

    for path in sorted((RAW_DIR / official).glob("*/*/traj_data.json")):
        d = json.load(open(path, encoding="utf-8"))
        actions = ["_".join([h["discrete_action"]["action"]] + h["discrete_action"]["args"])
                   for h in d["plan"]["high_pddl"] if h["discrete_action"]["action"] != "NoOp"]

        for ann in d["turk_annotations"]["anns"]:
            steps = [s.strip() for s in ann["high_descs"]]
            key = (d["task_id"], tuple(steps))
            if not ann["task_desc"].strip() or not all(steps):
                stats["empty"] += 1
            elif len(steps) != len(actions):
                stats["misaligned"] += 1
            elif key in seen:
                stats["duplicate"] += 1
            else:
                seen.add(key)
                samples.append({"task_id": d["task_id"],
                                "task_type": d["task_type"],
                                "task_desc": ann["task_desc"].strip(),
                                "steps": steps,
                                "actions": actions})

    with open(OUT_DIR / f"{role}.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report[role] = {"official": official,
                    **stats,
                    "kept": len(samples),
                    "task_ids": {s["task_id"] for s in samples}}
    print(role, {k: v for k, v in report[role].items() if k != "task_ids"})

ids = {r: report[r].pop("task_ids") for r in report}
report["overlap"] = {"train&valid": len(ids["train"] & ids["valid"]),
                     "train&test": len(ids["train"] & ids["test"]),
                     "valid&test": len(ids["valid"] & ids["test"])}
print("overlap", report["overlap"])
json.dump(report, open(OUT_DIR / "extract_report.json", "w"), indent=2)