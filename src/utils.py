import random
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

def load_config(name: str = "base.yaml") -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve(rel_path: str) -> Path:
    return ROOT / rel_path

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        print("[set_seed] numpy가 설치되어 있지 않아 numpy 시드가 고정되지 않았습니다.")
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        print("[set_seed] torch가 설치되어 있지 않아 torch 시드가 고정되지 않았습니다.")
        
    