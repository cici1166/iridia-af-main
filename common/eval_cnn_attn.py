# common/eval_cnn_attn.py
import sys, os, csv
print("[Eval] file:", os.path.abspath(__file__), flush=True)
print("[Eval] cwd :", os.getcwd(), flush=True)
print("[Eval] py  :", sys.executable, flush=True)

sys.path.append('.')
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

from common.model_cnn_attn import CNNWithAttention
from common.dataset_h5 import ECGH5Dataset

def okf(p): return os.path.isfile(p)

@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        prob = torch.sigmoid(model(xb))
        ys.append(yb.detach().cpu().numpy()); ps.append(prob.detach().cpu().numpy())
    if len(ys) == 0:
        return float('nan'), float('nan'), float('nan')
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.int64)
    acc = accuracy_score(y, pred); f1 = f1_score(y, pred)
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float('nan')
    return acc, f1, auc

def append_csv(path, row):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","split","accuracy","f1","auc"])
        if write_header: w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="dataset/processed_data.h5")
    ap.add_argument("--val_csv", default="splits/val.csv")
    ap.add_argument("--test_csv", default="splits/test.csv")
    ap.add_argument("--ckpt", default="best_model_cnn_attn.pt")
    ap.add_argument("--results_csv", default="results_val_test.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width_mult", type=float, default=None)
    ap.add_argument("--fc_dim", type=int, default=None)
    ap.add_argument("--attn_dim", type=int, default=None)
    args = ap.parse_args()

    print("[Eval] starting…", flush=True)
    print(f"[Eval] ckpt={args.ckpt}", flush=True)

    # 如果关键文件缺失，这里会明确报出来
    missing = []
    for p, name in [(args.ckpt,"ckpt"), (args.h5,"h5"), (args.val_csv,"val_csv"), (args.test_csv,"test_csv")]:
        if not okf(p):
            missing.append(f"{name} not found: {os.path.abspath(p)}")
    if missing:
        print("[Eval][ERROR]\n  " + "\n  ".join(missing), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    width_mult = args.width_mult if args.width_mult is not None else ckpt.get("width_mult", 1.0)
    fc_dim     = args.fc_dim     if args.fc_dim     is not None else ckpt.get("fc_dim", 128)
    attn_dim   = args.attn_dim   if args.attn_dim   is not None else ckpt.get("attn_dim", 128)
    print(f"[Eval] width_mult={width_mult} fc_dim={fc_dim} attn_dim={attn_dim}", flush=True)

    model = CNNWithAttention(width_mult=width_mult, fc_dim=fc_dim, attn_dim=attn_dim).to(device)
    model.load_state_dict(ckpt["model"])

    pin = True if torch.cuda.is_available() else False
    val_ds  = ECGH5Dataset(args.h5, args.val_csv,  normalize=False)
    test_ds = ECGH5Dataset(args.h5, args.test_csv, normalize=False)
    print(f"[Eval] #val={len(val_ds)}  #test={len(test_ds)}", flush=True)
    if len(val_ds) == 0 or len(test_ds) == 0:
        print("[Eval][ERROR] empty dataset (val or test).", flush=True)
        return

    val_loader  = DataLoader(val_ds,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=pin)

    for split, loader in [("Val", val_loader), ("Test", test_loader)]:
        acc, f1, auc = evaluate_split(model, loader, device)
        print(f"[CNN+Attn] {split}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}", flush=True)
        append_csv(args.results_csv, {
            "model": "cnn_attn", "split": split,
            "accuracy": f"{acc:.4f}", "f1": f"{f1:.4f}", "auc": f"{auc:.4f}",
        })
    print("[Eval] done.", flush=True)

if __name__ == "__main__":
    main()
