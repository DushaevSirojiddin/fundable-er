"""Build a fake holdout from the visible set: new input_ids, shuffled rows, same DB.

    python3 scripts/holdout_sim.py case-study scratch/holdout_sim
    CACHE_PATH=scratch/empty.sqlite uv run solve.py --db scratch/holdout_sim --input scratch/holdout_sim --out scratch/hs_out
    python3 case-study/grade.py scratch/hs_out --key scratch/holdout_sim --holdout

Checks that the solver does not depend on id prefixes or row order, and (with an empty cache) that a cold run
finishes inside the time limit. The remapped answer key is written only for grade.py; the solver never reads it.
"""
import csv
import random
import shutil
import sys
from pathlib import Path


def main(src: str, dst: str, seed: int = 11) -> None:
    src_p, dst_p = Path(src), Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for t in ("organizations", "people"):
        shutil.copy(src_p / f"existing_{t}.csv", dst_p / f"existing_{t}.csv")
        with open(src_p / f"input_{t}.csv", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        ids = [r["input_id"] for r in rows]
        new_ids = [f"h{t[0]}_{rng.randrange(16**6):06x}" for _ in ids]
        remap = dict(zip(ids, new_ids))
        rng.shuffle(rows)
        for r in rows:
            r["input_id"] = remap[r["input_id"]]
        for name in (f"input_{t}.csv", f"holdout_input_{t}.csv"):
            with open(dst_p / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        with open(src_p / f"answer_key_{t}.csv", newline="", encoding="utf-8-sig") as fh:
            key = list(csv.DictReader(fh))
        for r in key:
            r["input_id"] = remap[r["input_id"]]
            r["dup_of"] = remap.get(r["dup_of"], "")
        rng.shuffle(key)
        with open(dst_p / f"answer_key_holdout_{t}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(key[0].keys()))
            w.writeheader()
            w.writerows(key)
    print(f"wrote {dst_p}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
