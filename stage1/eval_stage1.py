#!/usr/bin/env python3
"""Evaluate Stage-1 models on the held-out TEST seed block (9.0M+).

Produces scorecard.json + two figures:
  rmse_bars.png    path-prediction RMSE: CV extrapolation vs GRU-corrected
  id_confusion.png threat-ID confusion: readout baseline / XGBoost / GRU
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_stage1 import (BASE, HORIZONS, GRUStage1, load_split, make_targets,
                          evaluate, cv_extrap, WINDOW, WARM)

CLASSES = ["quad", "fixed_wing", "bird"]
OUT = BASE


def main():
    d = load_split("test")
    print(f"test: {d['N']} episodes, K={d['K']}")
    corr, pmask, idm = make_targets(d)
    yv = np.repeat(d["cls"][:, None], d["K"], 1)

    # ---- path prediction -------------------------------------------------
    model = GRUStage1()
    model.load_state_dict(torch.load(BASE / "models" / "gru_stage1.pt",
                                     map_location="cpu"))
    acc_gru, rmses = evaluate(model, d, corr, pmask, idm)

    # per-scenario 2 s breakdown + steady-state (t>=2 s) variant
    scen = np.array(d["scen"])
    idx = np.arange(d["K"])
    steady = idm & (idx[None] >= 20)
    per_scen, steady_rmse = {}, []
    for hi, H in enumerate(HORIZONS):
        m = pmask[:, :, hi] & steady
        fut = np.clip(idx + H, 0, d["K"] - 1)
        base = cv_extrap(d["est_pos"], d["est_vel"], H)
        with torch.no_grad():
            _, path = model(torch.tensor(d["X"]))
        pred = base + path[:, :, hi, :].numpy() * 100.0
        eb = np.linalg.norm(base - d["truth_pos"][:, fut, :], axis=-1)[m]
        em = np.linalg.norm(pred - d["truth_pos"][:, fut, :], axis=-1)[m]
        steady_rmse.append((round(float(np.sqrt((eb**2).mean())), 2),
                            round(float(np.sqrt((em**2).mean())), 2)))
    for s in sorted(set(d["scen"])):
        sel = scen == s
        m = pmask[:, 1, :] & steady & sel[:, None]
        base = cv_extrap(d["est_pos"], d["est_vel"], HORIZONS[1])
        with torch.no_grad():
            _, path = model(torch.tensor(d["X"]))
        pred = base + path[:, :, 1, :].numpy() * 100.0
        fut = np.clip(idx + HORIZONS[1], 0, d["K"] - 1)
        eb = np.linalg.norm(base - d["truth_pos"][:, fut, :], axis=-1)[m]
        em = np.linalg.norm(pred - d["truth_pos"][:, fut, :], axis=-1)[m]
        per_scen[s] = dict(cv=round(float(np.sqrt((eb**2).mean())), 2),
                           gru=round(float(np.sqrt((em**2).mean())), 2),
                           n=int(m.sum()))

    # ---- threat ID -------------------------------------------------------
    # GRU per-frame (already have acc_gru); majority vote over t>=1 s for a
    # fair 'detector-level' number:
    with torch.no_grad():
        id_logits, _ = model(torch.tensor(d["X"]))
    id_pred = id_logits.argmax(-1).numpy()
    gru_acc_steady = float((id_pred[steady] == yv[steady]).mean())

    from xgboost import XGBClassifier
    clf = XGBClassifier()
    clf.load_model(str(BASE / "models" / "xgb_id.json"))
    from train_stage1 import window_features
    Fte = window_features(d)
    mte = d["valid"] & ~d["off"] & (idx[None] >= WARM + WINDOW)
    yte = np.concatenate([np.full(mte[n].sum(), d["cls"][n])
                          for n in range(d["N"])])
    xgb_pred = clf.predict(Fte[mte])
    xgb_acc = float((xgb_pred == yte).mean())
    win = Fte[mte]
    posts = win[:, 1:5] + win[:, 6:10] + win[:, 11:15]
    base_pred = posts.argmax(axis=1)
    base_acc = float((base_pred == yte).mean())
    cm_xgb = np.zeros((3, 3), int)
    for y, p in zip(yte, xgb_pred):
        if y < 3 and p < 3:
            cm_xgb[y, p] += 1
    cm_base = np.zeros((3, 3), int)
    for y, p in zip(yte, base_pred):
        if y < 3 and p < 3:
            cm_base[y, p] += 1
    per_cls = {}
    for c in range(3):
        sel = yte == c
        per_cls[CLASSES[c]] = dict(
            readout=round(float((base_pred[sel] == c).mean()), 3),
            xgb=round(float((xgb_pred[sel] == c).mean()), 3))
    gru_recall = {}
    for c in range(3):
        sel = steady & (yv == c)
        gru_recall[CLASSES[c]] = round(float((id_pred[sel] == c).mean()), 3)

    # ---- figures ----------------------------------------------------------
    labels = [f"{H*0.1:.0f}s" for H in HORIZONS]
    x = np.arange(len(HORIZONS))
    cv_all = [r[0] for r in rmses]
    gru_all = [r[1] for r in rmses]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    b1 = ax.bar(x - 0.18, cv_all, 0.36, label="CV extrapolation (EKF state)",
                color="#9aa7b5")
    b2 = ax.bar(x + 0.18, gru_all, 0.36, label="GRU-corrected (Stage-1)",
                color="#2563eb")
    ax.bar_label(b1, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, fmt="%.1f", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("position RMSE (m)")
    ax.set_title(f"Path prediction on test seeds ({d['N']} episodes)")
    ax.legend()
    fig.savefig(OUT / "rmse_bars.png", dpi=140)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    for ax, cmx, name in ((axes[0], cm_base, f"Readout baseline ({base_acc:.0%})"),
                          (axes[1], cm_xgb, f"XGBoost ({xgb_acc:.0%})"),
                          (axes[2], None, f"GRU multi-task ({acc_gru:.0%})")):
        if cmx is None:
            cmx = np.zeros((3, 3), int)
            for n in range(d["N"]):
                sel = steady[n]
                for y, p in zip(yv[n][sel], id_pred[n][sel]):
                    if y < 3:
                        cmx[y, p] += 1
        im = ax.imshow(cmx, cmap="Blues")
        ax.set_xticks(range(3), CLASSES, rotation=30)
        ax.set_yticks(range(3), CLASSES)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, cmx[a, b], ha="center", va="center",
                        color="white" if cmx[a, b] > cmx.max() / 2 else "black",
                        fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true class")
    fig.suptitle("Threat ID confusion — held-out test seeds", fontsize=11)
    fig.savefig(OUT / "id_confusion.png", dpi=140)

    # ---- scorecard ---------------------------------------------------------
    scorecard = {
        "test_episodes": d["N"],
        "seed_block": "9_000_000 - 9_000_079 (never seen in train/val)",
        "path_prediction": {
            "rmse_m_all_frames": {f"{H*0.1:.0f}s": dict(
                cv_extrap=round(r[0], 2), gru_corrected=round(r[1], 2),
                improvement_pct=round(100 * (r[0] - r[1]) / r[0], 1))
                for (r, H) in zip(rmses, HORIZONS)},
            "rmse_m_steady_t2s": {f"{H*0.1:.0f}s": dict(
                cv_extrap=s[0], gru_corrected=s[1])
                for (s, H) in zip(steady_rmse, HORIZONS)},
            "per_scenario_2s": per_scen,
        },
        "threat_id": {
            "readout_baseline_acc": round(base_acc, 4),
            "xgboost_acc": round(xgb_acc, 4),
            "gru_frame_acc_t1s": round(acc_gru, 4),
            "gru_frame_acc_steady": round(gru_acc_steady, 4),
            "per_class_recall": per_cls,
            "gru_per_class_recall_steady": gru_recall,
        },
        "note": "masks: track_valid & not off(>500 m) & warm-up excluded; "
                "identical rules at train and test",
    }
    (OUT / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    print(json.dumps(scorecard, indent=2)[:2400])


if __name__ == "__main__":
    main()
