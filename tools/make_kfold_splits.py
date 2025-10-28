
import json, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

d = np.load(args.npz, allow_pickle=True)
seq = None
for k in ["SEQ_NAME","seq_name","seq"]:
    if k in d.files:
        seq = d[k]
        break
if seq is None:
    # Align with IMUFrames fallback naming: seq_0, seq_1, ...
    seq = np.array([f"seq_{i}" for i in range(d[list(d.files)[0]].shape[0])])

uniq = np.unique(seq)
rng = np.random.default_rng(args.seed)
rng.shuffle(uniq)

folds = []
for i in range(args.k):
    val = uniq[i::args.k]
    tr  = np.array([s for s in uniq if s not in val])
    folds.append({"train": tr.tolist(), "val": val.tolist()})

with open(args.out_json, "w", encoding="utf-8") as f:
    json.dump({"folds": folds}, f, indent=2, ensure_ascii=False)
print("Saved", args.out_json)
