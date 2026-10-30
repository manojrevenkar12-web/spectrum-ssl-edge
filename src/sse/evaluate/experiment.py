"""The experiment: one command builds the data, trains every model, and measures them.

Questions it answers (docs/REPORT.md is generated from the answers):
  Q1 Does contrastive pretraining help when labels are scarce, and does the gain vanish with many labels?
  Q2 Does pretraining on unlabelled *field* recordings make the classifier hold up under shift, including
     shifts no pretraining data contained (oversampling, the held-out domain)?
  Q3 Does temperature scaling keep a model calibrated under shift, and do ensembles do better
     (Ovadia et al., NeurIPS 2019)?
  Q4 Which score finds classes never seen in training, near ones and far ones?
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax

from .. import config
from ..models.encoder import Classifier, count_macs
from ..signals.dataset import Windows, make_windows
from ..signals.modulate import KNOWN, NOVEL
from ..train.fit import linear_probe, predict, pretrain_contrastive, train_classifier
from . import metrics as M

K = len(KNOWN)


def stratified(w: Windows, per_class: int, seed: int) -> Windows:
    rng = np.random.default_rng(10_000 + seed)
    idx = []
    for c in range(K):
        pool = np.nonzero(w.y == c)[0]
        idx.append(rng.permutation(pool)[:per_class])
    return w.subset(np.sort(np.concatenate(idx)))


def build_data(p: dict[str, Any], length: int) -> dict[str, Any]:
    doms = config.domains()
    held = config.heldout_domain()
    lab = doms["lab"]
    ecfg = config.load("experiment.yaml")
    data: dict[str, Any] = {
        "train": make_windows(KNOWN, p["train_per_class"], lab, length, seed=101),
        "val": make_windows(KNOWN, p["val_per_class"], lab, length, seed=102),
        "test": {},
        "novel": {},
        "pools": {},
    }
    test_domains = dict(doms)
    if held is not None:
        test_domains["heldout"] = held
    for i, (name, d) in enumerate(test_domains.items()):
        data["test"][name] = make_windows(KNOWN, p["test_per_class"], d, length, seed=200 + i)
    for i, name in enumerate(ecfg["openset_domains"]):
        data["novel"][name] = make_windows(NOVEL, p["novel_per_class"], doms[name], length, seed=300 + i)
    for j, (pool_name, sources) in enumerate(ecfg["ssl_pools"].items()):
        per_source = p["ssl_pool"] // len(sources)
        parts = [
            make_windows(KNOWN, max(1, per_source // K), doms[s], length, seed=400 + 10 * j + i)
            for i, s in enumerate(sources)
        ]
        data["pools"][pool_name] = Windows.concat(parts)
    return data


def _summ(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def run(profile: str = "full", out_dir: Path | None = None, log=print) -> dict[str, Any]:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ecfg = config.load("experiment.yaml")
    p = ecfg["profiles"][profile]
    tcfg = ecfg["train"]
    length = ecfg["length"]
    seeds = ecfg["seeds"]
    out_dir = out_dir or config.ROOT / "runs" / profile
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log(f"[data] building ({profile})")
    data = build_data(p, length)
    domains = list(data["test"])
    dev_domains = [d for d in domains if d != "heldout"]

    models: dict[str, Classifier] = {}
    full = max(p["budgets"])

    log("[train] supervised baselines")
    for b in p["budgets"]:
        for s in seeds:
            sub = stratified(data["train"], b, s)
            models[f"supervised/{b}/{s}"] = train_classifier(
                sub, K, tcfg, ecfg["augment"]["light"], p["supervised_epochs"], p["min_steps"], seed=s
            )

    ssl_curves = {}
    for pool_name, pool in data["pools"].items():
        log(f"[ssl] contrastive pretraining on the {pool_name} pool ({len(pool)} windows)")
        enc, losses = pretrain_contrastive(pool, tcfg, ecfg["augment"]["strong"], p["ssl_epochs"], seed=0)
        ssl_curves[pool_name] = losses[:: max(1, len(losses) // 100)]
        torch.save(enc.state_dict(), out_dir / "models" / f"encoder-{pool_name}.pt")
        for b in p["budgets"]:
            for s in seeds:
                sub = stratified(data["train"], b, s)
                models[f"ssl-{pool_name}-probe/{b}/{s}"] = linear_probe(enc, sub, K, tcfg["probe_steps"], seed=s)
                models[f"ssl-{pool_name}-finetune/{b}/{s}"] = train_classifier(
                    sub, K, tcfg, ecfg["augment"]["light"], p["supervised_epochs"], p["min_steps"], seed=s, encoder=enc
                )

    log(f"[eval] {len(models)} models x {len(domains)} domains")
    preds = {name: {d: predict(m, data["test"][d]) for d in domains} for name, m in models.items()}
    methods = sorted({n.split("/")[0] for n in models})

    # Q1/Q2: label efficiency and robustness
    label_eff: dict[str, Any] = {}
    for meth in methods:
        label_eff[meth] = {}
        for b in p["budgets"]:
            label_eff[meth][str(b)] = {
                d: _summ([M.accuracy(preds[f"{meth}/{b}/{s}"][d].logits, data["test"][d].y) for s in seeds])
                for d in domains
            }

    def dev_mean(meth: str, b: int) -> float:
        return float(np.mean([label_eff[meth][str(b)][d]["mean"] for d in dev_domains]))

    deployed_method = max(methods, key=lambda m: dev_mean(m, full))
    deployed = f"{deployed_method}/{full}/0"
    log(f"[select] deployed model: {deployed} (best mean accuracy over dev domains at full labels)")

    by_snr = {
        meth: {
            d: M.accuracy_by_snr(preds[f"{meth}/{full}/0"][d].logits, data["test"][d].y, data["test"][d].meta["snr_db"])
            for d in domains
        }
        for meth in methods
    }

    # Q3: calibration under shift. Temperature fitted on the bench validation split only.
    val_logits = {
        n: predict(models[n], data["val"]).logits
        for n in models
        if n.endswith(f"/{full}/0") or n.startswith(f"supervised/{full}/")
    }
    ens_names = [f"supervised/{full}/{s}" for s in seeds[: p["ensemble"]]]
    cal_models: dict[str, Any] = {}

    def probs_of(name: str, d: str, temp: float = 1.0) -> np.ndarray:
        return softmax(preds[name][d].logits / temp, axis=1)

    t_sup = M.fit_temperature(val_logits[f"supervised/{full}/0"], data["val"].y)
    t_dep = M.fit_temperature(val_logits[deployed], data["val"].y)
    ens_val = np.mean([softmax(val_logits[n], axis=1) for n in ens_names], axis=0)
    t_ens = M.fit_temperature(np.log(ens_val + 1e-12), data["val"].y)
    variants = {
        "supervised": lambda d: probs_of(f"supervised/{full}/0", d),
        "supervised + temperature": lambda d: probs_of(f"supervised/{full}/0", d, t_sup),
        f"ensemble of {len(ens_names)}": lambda d: np.mean([probs_of(n, d) for n in ens_names], axis=0),
        f"ensemble of {len(ens_names)} + temperature": lambda d: softmax(
            np.log(np.mean([probs_of(n, d) for n in ens_names], axis=0) + 1e-12) / t_ens, axis=1
        ),
        f"{deployed_method}": lambda d: probs_of(deployed, d),
        f"{deployed_method} + temperature": lambda d: probs_of(deployed, d, t_dep),
    }
    for vname, fn in variants.items():
        cal_models[vname] = {}
        for d in domains:
            pr = fn(d)
            y = data["test"][d].y
            s = M.calibration_summary(pr, y)
            rc = M.risk_coverage(pr, y)
            s.update(aurc=rc["aurc"], risk_at_80=rc["risk_at_80"])
            cal_models[vname][d] = s
    reliability = {
        v: {d: M.reliability(variants[v](d), data["test"][d].y) for d in domains}
        for v in ("supervised + temperature", f"{deployed_method} + temperature")
    }
    risk_curves = {
        v: {d: M.risk_coverage(variants[v](d), data["test"][d].y)["curve"] for d in domains} for v in variants
    }

    # Q4: open set
    groups = ecfg["novel_groups"]
    openset: dict[str, Any] = {}
    for mname in (f"supervised/{full}/0", deployed):
        mod = models[mname]
        tr = predict(mod, data["train"])
        maha = M.Mahalanobis(tr.embeddings, data["train"].y, K)
        knn = M.KNN(tr.embeddings)
        openset[mname] = {}
        for d in data["novel"]:
            kn = preds[mname][d]
            nv = predict(mod, data["novel"][d])
            scores = {
                "max softmax": (M.msp_score(kn.logits), M.msp_score(nv.logits)),
                "energy": (M.energy_score(kn.logits), M.energy_score(nv.logits)),
                "mahalanobis": (maha.score(kn.embeddings), maha.score(nv.embeddings)),
                "knn": (knn.score(kn.embeddings), knn.score(nv.embeddings)),
            }
            openset[mname][d] = {}
            for sname, (si, so) in scores.items():
                row: dict[str, Any] = {}
                for gname, members in groups.items():
                    sel = np.isin(data["novel"][d].name, members)
                    row[gname] = {"auroc": M.auroc(si, so[sel]), "fpr95": M.fpr_at_tpr(si, so[sel])}
                for c in NOVEL:
                    sel = data["novel"][d].name == c
                    row[c] = {"auroc": M.auroc(si, so[sel])}
                openset[mname][d][sname] = row

    confusions = {d: M.confusion(preds[deployed][d].logits, data["test"][d].y, K) for d in domains}

    torch.save(models[deployed].state_dict(), out_dir / "models" / "deployed.pt")
    torch.save(models[f"supervised/{full}/0"].state_dict(), out_dir / "models" / "supervised.pt")
    np.savez_compressed(out_dir / "calibration.npz", x=data["val"].x[:512], mask=data["val"].mask[:512])
    np.savez_compressed(
        out_dir / "lab_test.npz", x=data["test"]["lab"].x, mask=data["test"]["lab"].mask, y=data["test"]["lab"].y
    )

    results = {
        "profile": profile,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "classes": {"known": KNOWN, "novel": NOVEL, "novel_groups": groups},
        "domains": domains,
        "budgets": p["budgets"],
        "seeds": seeds,
        "sizes": {
            "train": len(data["train"]),
            "val": len(data["val"]),
            "test_per_domain": {d: len(w) for d, w in data["test"].items()},
            "novel_per_domain": {d: len(w) for d, w in data["novel"].items()},
            "pools": {k: len(v) for k, v in data["pools"].items()},
        },
        "model": {
            "parameters": sum(t.numel() for t in models[deployed].parameters()),
            "macs_per_window": count_macs(length),
            "length": length,
        },
        "deployed": {
            "name": deployed,
            "method": deployed_method,
            "rule": "highest mean accuracy over the dev domains at full labels",
        },
        "label_efficiency": label_eff,
        "accuracy_by_snr": by_snr,
        "ssl_loss": ssl_curves,
        "temperatures": {"supervised": t_sup, "ensemble": t_ens, deployed_method: t_dep},
        "calibration": cal_models,
        "reliability": reliability,
        "risk_coverage": risk_curves,
        "openset": openset,
        "confusion": confusions,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    log(f"[done] {results['seconds']} s")
    return results
