#!/usr/bin/env python3
"""Train Stage-1 models on the SkyGym corpus.

Rung 1: XGBoost threat-ID on engineered 1 s window features.
Rung 2: GRU multi-task - threat ID + corrections to Stone Soup CV
        extrapolation at horizons 1 s / 2 s / 4 s.

Inputs are strictly legitimate: tracker state + observation summaries.
Truth is used ONLY as targets. Train/val splits are seed-blocked.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Project root: override with SKYGYM_PROJECT_ROOT (see STAGE1_RUNBOOK.md §1)
PROJECT = os.environ.get("SKYGYM_PROJECT_ROOT", "/home/z/my-project")
BASE = Path(PROJECT) / "download" / "stage1"
HORIZONS = [10, 20, 40]          # frames -> 1 s, 2 s, 4 s
WARM = 5                         # skip t < 0.5 s
WINDOW = 10                      # XGBoost look-back (1 s)
torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------------------------ #
def load_split(split):
    eps = [json.loads(l) for l in open(BASE / "corpus" / f"{split}.jsonl")]
    N = len(eps)
    K = max(e["K"] for e in eps)
    X = np.zeros((N, K, 31), dtype=np.float32)
    valid = np.zeros((N, K), dtype=bool)
    off = np.zeros((N, K), dtype=bool)
    est_pos = np.zeros((N, K, 3), dtype=np.float64)
    est_vel = np.zeros((N, K, 3), dtype=np.float64)
    truth_pos = np.zeros((N, K, 3), dtype=np.float64)
    cls = np.zeros(N, dtype=np.int64)
    scen = []
    for n, e in enumerate(eps):
        k = e["K"]
        X[n, :k] = np.array(e["feats"], dtype=np.float32)
        valid[n, :k] = np.array(e["track_valid"]) > 0
        off[n, :k] = np.array(e["off"]) > 0
        ep = np.array(e["est_pos"]);  est_pos[n, :k] = np.nan_to_num(ep)
        ev = np.array(e["est_vel"]);  est_vel[n, :k] = np.nan_to_num(ev)
        truth_pos[n, :k] = np.array(e["truth_pos"])
        cls[n] = e["class_idx"]
        scen.append(e["scenario"])
    return dict(X=X, valid=valid, off=off, est_pos=est_pos, est_vel=est_vel,
                truth_pos=truth_pos, cls=cls, scen=scen, K=K, N=N)


# ------------------------------------------------------------------ #
# Rung 1: XGBoost threat ID on window features
# ------------------------------------------------------------------ #
def window_features(d):
    """(N, K, 31) -> (N, K, F) engineered features over the last WINDOW frames."""
    X, N, K, _ = d["X"], d["N"], d["K"], None
    F = []
    cols = dict(rad_n=8, rad_snr=9, rad_d=11, rad_p=(12, 16),
                eo_n=16, eo_px=17, eo_d=19, eo_p=(20, 24),
                rf_n=24, rf_az=25, rf_p=(26, 30))
    for n in range(N):
        rows = []
        for k in range(K):
            lo = max(0, k - WINDOW)
            w = X[n, lo:k + 1]                      # (W, 31)
            f = []
            for c in ("rad", "eo", "rf"):
                f.append(w[:, cols[f"{c}_n"]].mean())
                f.extend(w[:, cols[f"{c}_p"][0]:cols[f"{c}_p"][1]].mean(axis=0))
            f.append(w[:, cols["rad_snr"]][w[:, cols["rad_n"]] > 0].mean()
                     if (w[:, cols["rad_n"]] > 0).any() else 0.0)
            f.append(w[:, cols["eo_px"]][w[:, cols["eo_n"]] > 0].mean()
                     if (w[:, cols["eo_n"]] > 0).any() else 0.0)
            f.append(w[:, cols["rf_az"]][w[:, cols["rf_n"]] > 0].mean()
                     if (w[:, cols["rf_n"]] > 0).any() else 0.0)
            f.append(w[:, cols["rad_d"]].min())
            f.append(w[:, cols["eo_d"]].min())
            # kinematics from tracker velocity (m/s)
            vel = w[:, 3:6] * 30.0
            spd = np.linalg.norm(vel, axis=1)
            f += [spd.mean(), spd.std(), spd.max() - spd.min(), vel[:, 2].mean(),
                  vel[:, 0].std(), vel[:, 1].std()]
            f += [w[-1, 6] * 200.0, w[-1, 7]]       # maturity, staleness
            rows.append(f)
        F.append(rows)
        if n % 100 == 0:
            print(f"  winfeat ep{n}/{N}", flush=True)
    return np.array(F, dtype=np.float32)


def train_xgb(dtr, dva):
    from xgboost import XGBClassifier
    from sklearn.metrics import recall_score, confusion_matrix
    print("[rung1] building window features ...", flush=True)
    Ftr, Fva = window_features(dtr), window_features(dva)
    mtr = dtr["valid"] & ~dtr["off"] & (np.arange(dtr["K"])[None] >= WARM + WINDOW)
    mva = dva["valid"] & ~dva["off"] & (np.arange(dva["K"])[None] >= WARM + WINDOW)
    ytr = np.concatenate([np.full(mtr[n].sum(), dtr["cls"][n]) for n in range(dtr["N"])])
    yva = np.concatenate([np.full(mva[n].sum(), dva["cls"][n]) for n in range(dva["N"])])
    Xtr, Xva = Ftr[mtr], Fva[mva]
    # class balance
    cnt = np.bincount(ytr, minlength=4)
    wtr = np.array([len(ytr) / (4 * cnt[y]) for y in ytr])
    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                        tree_method="hist", n_jobs=2, eval_metric="mlogloss")
    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=wtr)
    print(f"[rung1] fit {time.time()-t0:.0f}s  train rows {len(ytr)}", flush=True)
    pred = clf.predict(Xva)
    acc = float((pred == yva).mean())
    rec = recall_score(yva, pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yva, pred, labels=[0, 1, 2]).tolist()
    # posterior-readout baseline on the SAME frames
    # window-feature layout: [rad_n, rad_p x4, eo_n, eo_p x4, rf_n, rf_p x4, ...]
    win = Fva
    posts = win[:, 1:5] + win[:, 6:10] + win[:, 11:15]
    base_pred = posts.argmax(axis=1)
    base_acc = float((base_pred == yva).mean())
    out = dict(acc=round(acc, 4), per_class_recall=[round(float(r), 4) for r in rec],
               confusion_012=cm, readout_baseline_acc=round(base_acc, 4),
               n_val_frames=int(len(yva)))
    print("[rung1] val:", json.dumps(out), flush=True)
    clf.save_model(str(BASE / "models" / "xgb_id.json"))
    (BASE / "models" / "xgb_id_val.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ #
# Rung 2: GRU multi-task
# ------------------------------------------------------------------ #
class GRUStage1(nn.Module):
    def __init__(self, din=31, h=64, layers=2, n_class=4, n_h=3):
        super().__init__()
        self.gru = nn.GRU(din, h, num_layers=layers, batch_first=True,
                          dropout=0.1)
        self.id_head = nn.Linear(h, n_class)
        self.path_head = nn.Linear(h, n_h * 3)
        self.n_h = n_h

    def forward(self, x):
        z, _ = self.gru(x)
        return self.id_head(z), self.path_head(z).view(x.shape[0], x.shape[1],
                                                       self.n_h, 3)


def cv_extrap(est_pos, est_vel, H, dt=0.1):
    return est_pos + est_vel * (H * dt)


def make_targets(d):
    N, K = d["N"], d["K"]
    corr = np.zeros((N, K, len(HORIZONS), 3), dtype=np.float32)
    pmask = np.zeros((N, K, len(HORIZONS)), dtype=bool)
    idx = np.arange(K)
    idm = d["valid"] & ~d["off"] & (idx[None] >= WARM)
    for hi, H in enumerate(HORIZONS):
        fut = np.clip(idx + H, 0, K - 1)
        tgt = d["truth_pos"][:, fut, :] - cv_extrap(d["est_pos"], d["est_vel"], H)
        ok = idm & (idx + H <= K - 1)
        corr[:, :, hi, :] = np.clip(tgt / 100.0, -5, 5)
        pmask[:, :, hi] = ok
    return corr, pmask, idm


def evaluate(model, d, corr, pmask, idm):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(d["X"])
        id_logits, path = model(xt)
    id_pred = id_logits.argmax(-1).numpy()
    yv = np.repeat(d["cls"][:, None], d["K"], 1)
    id_acc = float((id_pred[idm] == yv[idm]).mean())
    rmses = []
    for hi, H in enumerate(HORIZONS):
        m = pmask[:, :, hi]
        base = cv_extrap(d["est_pos"], d["est_vel"], H)
        pred = base + path[:, :, hi, :].numpy() * 100.0
        err_b = np.linalg.norm(base - d["truth_pos"][:, np.clip(
            np.arange(d["K"]) + H, 0, d["K"] - 1), :], axis=-1)[m]
        err_m = np.linalg.norm(pred - d["truth_pos"][:, np.clip(
            np.arange(d["K"]) + H, 0, d["K"] - 1), :], axis=-1)[m]
        rmses.append((float(np.sqrt((err_b ** 2).mean())),
                      float(np.sqrt((err_m ** 2).mean()))))
    return id_acc, rmses


def train_gru(dtr, dva):
    corr_tr, pm_tr, idm_tr = make_targets(dtr)
    corr_va, pm_va, idm_va = make_targets(dva)
    Xt = torch.tensor(dtr["X"])
    ct = torch.tensor(corr_tr)
    ytr_ep = torch.tensor(dtr["cls"])
    idm_t = torch.tensor(idm_tr)
    pm_t = torch.tensor(pm_tr)
    model = GRUStage1()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    hub = nn.HuberLoss(delta=1.0)
    ce = nn.CrossEntropyLoss()
    best, best_state = 1e9, None
    N = dtr["N"]
    for ep in range(40):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for b in range(0, N, 16):
            bi = perm[b:b + 16]
            id_logits, path = model(Xt[bi])
            lid = ce(id_logits[idm_t[bi]], ytr_ep[bi][:, None].expand(
                -1, dtr["K"])[idm_t[bi]])
            lpath = sum(hub(path[:, :, hi][pm_t[bi][:, :, hi]],
                            ct[bi][:, :, hi][pm_t[bi][:, :, hi]])
                        for hi in range(len(HORIZONS))) / len(HORIZONS)
            loss = 0.5 * lid + 1.0 * lpath
            opt.zero_grad();  loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        acc_va, rmses = evaluate(model, dva, corr_va, pm_va, idm_va)
        score = rmses[1][1] + 20.0 * (1.0 - acc_va)   # 2 s RMSE + id term
        if score < best:
            best = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[rung2] ep{ep:02d} loss {tot/((N+15)//16):.3f} "
              f"val id {acc_va:.3f} | RMSE CV->GRU m: "
              + " ".join(f"{H*0.1:.0f}s {b:6.2f}->{m:6.2f}"
                         for (b, m), H in zip(rmses, HORIZONS)), flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), BASE / "models" / "gru_stage1.pt")
    return model


def main():
    print("loading corpus ...", flush=True)
    dtr, dva = load_split("train"), load_split("val")
    print(f"train {dtr['N']} eps / val {dva['N']} eps, K={dtr['K']}", flush=True)
    xgb_out = train_xgb(dtr, dva)
    model = train_gru(dtr, dva)
    corr_va, pm_va, idm_va = make_targets(dva)
    acc_va, rmses = evaluate(model, dva, corr_va, pm_va, idm_va)
    print("FINAL val:", json.dumps(dict(id_acc=acc_va, rmse=rmses), default=float))
    (BASE / "models" / "gru_val.json").write_text(json.dumps(
        dict(id_acc=acc_va, rmse=rmses), default=float, indent=2))


if __name__ == "__main__":
    main()
