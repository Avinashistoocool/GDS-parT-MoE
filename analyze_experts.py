"""
analyze_experts.py — what physics did each expert learn?

For a trained GDS-ParT-MoE checkpoint this records, for every real particle in
every MoE layer, the router's top-1 expert and full routing distribution, then
relates them to physics quantities of that particle and its jet:

  particle   type (ch. hadron / neu. hadron / photon / e / mu), charge,
             ln pT, pT fraction z = pT/pT_jet, angular distance dR to the jet axis,
             transverse impact parameter |d0| (tanh-compressed, as in the inputs),
             pT rank inside the jet, distance to the nearest neighbour dR_nn,
             and local kT to that neighbour (ln kT, the Lund-plane variable)
  jet        truth class (H->bb, t->bqq', QCD, ...), constituent multiplicity

Outputs (in --out):
  routing_<layer>.png       P(expert | particle type) and mean physics per expert
  class_usage.png           P(expert | jet class) per layer
  mutual_information.png    normalised MI(expert ; variable) per layer
  knockout.png              accuracy change per jet class when one expert is removed
  report.json, REPORT.md    all the numbers plus a plain-language summary

    python analyze_experts.py --ckpt checkpoints/moe8/best.pt --cache_dir D:/JetClass/cache \
        --val_cache val_val_50000_128 --jets 50000 --out analysis/moe8
    python analyze_experts.py --ckpt checkpoints/smoke/best.pt --synthetic   # wiring test
"""
import argparse
import json
import os

import numpy as np
import torch

from moe import GDSParTMoE, GDSParTMoEConfig
from train_moe import CLASS_LIST, prepare_batch, real_loader, synthetic_loader

PTYPES = ["ch.had", "neu.had", "photon", "electron", "muon"]
# continuous variables -> (label, quantile bins for MI)
CONT = [("ln_pt", "ln pT"), ("z", "pT / pT_jet"), ("dR", "dR to axis"), ("abs_d0", "|tanh d0|"),
        ("rank", "pT rank"), ("dR_nn", "dR to nearest"), ("ln_kt_nn", "ln kT to nearest")]


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GDSParTMoEConfig(**ck["config"])
    model = GDSParTMoE(cfg)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), cfg


def particle_physics(feats, vectors, mask):
    """Undo the input normalisation (data/jetclass_cache.py) and add neighbour
    variables computed from the 4-vectors. Returns dict of flat arrays over
    real particles, in the same order as `x[mask]`."""
    f = feats.float()
    out = {
        "ptype": f[..., 8:13].argmax(-1),
        "charge": f[..., 7].round(),
        "ln_pt": f[..., 0] / 0.7 + 1.7,
        "z": torch.exp(f[..., 2] / 0.7 - 4.7),
        "dR": f[..., 4] / 4.0 + 0.2,
        "abs_d0": f[..., 13].abs(),
    }
    B, N = mask.shape
    if vectors is not None:
        v = vectors.float()
        pt = torch.hypot(v[..., 0], v[..., 1]).clamp(min=1e-8)
        rap = 0.5 * torch.log((v[..., 3] + v[..., 2]).clamp(min=1e-8) / (v[..., 3] - v[..., 2]).clamp(min=1e-8))
        phi = torch.atan2(v[..., 1], v[..., 0])
        dphi = torch.remainder(phi[:, :, None] - phi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
        d = torch.sqrt((rap[:, :, None] - rap[:, None, :]) ** 2 + dphi ** 2)
        bad = ~(mask[:, :, None] & mask[:, None, :]) | torch.eye(N, dtype=torch.bool, device=mask.device)
        d = d.masked_fill(bad, float("inf"))
        dnn, j = d.min(-1)
        pt_j = pt.gather(1, j)
        out["dR_nn"] = torch.where(torch.isinf(dnn), torch.zeros_like(dnn), dnn)
        out["ln_kt_nn"] = torch.log((torch.minimum(pt, pt_j) * out["dR_nn"]).clamp(min=1e-4))
        order = torch.argsort(pt.masked_fill(~mask, -1), dim=1, descending=True)
        out["rank"] = torch.empty_like(order).scatter_(1, order, torch.arange(N, device=mask.device).expand(B, N)).float()
    else:
        out["dR_nn"] = torch.zeros(B, N, device=mask.device)
        out["ln_kt_nn"] = torch.zeros(B, N, device=mask.device)
        out["rank"] = torch.arange(N, device=mask.device).float().expand(B, N)
    return {k: t[mask].cpu().numpy() for k, t in out.items()}


@torch.no_grad()
def collect(model, loader, device, max_jets, max_pairs):
    layers = [i for i, _ in model.moe_layers()]
    model.set_record(True)
    rec = {"top": {i: [] for i in layers}, "probs": {i: [] for i in layers}}
    phys, jet_label, jet_of, jet_scores, n_jets = [], [], [], [], 0
    for batch in loader:
        feats, pairs, mask, y, vectors = prepare_batch(batch, device)
        if n_jets + len(y) > max_jets:
            k = max_jets - n_jets
            feats, pairs, mask, y = feats[:k], pairs[:k], mask[:k], y[:k]
            vectors = vectors[:k] if vectors is not None else None
        N = feats.shape[1]
        micro = max(1, max_pairs // (N * N))
        for s in range(0, len(y), micro):
            sl = slice(s, s + micro)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                jet_scores.append(model(feats[sl], pairs[sl], mask=mask[sl]).float().softmax(-1).cpu().numpy())
            for i, m in model.moe_layers():
                rec["top"][i].append(m.last["top"][:, 0].numpy())
                rec["probs"][i].append(m.last["probs"].numpy().astype(np.float16))
        phys.append(particle_physics(feats, vectors, mask))
        counts = mask.sum(1).cpu().numpy()
        jet_label.append(y.cpu().numpy())
        jet_of.append(np.repeat(np.arange(n_jets, n_jets + len(y)), counts))
        n_jets += len(y)
        if n_jets >= max_jets:
            break
    model.set_record(False)
    P = {k: np.concatenate([p[k] for p in phys]) for k in phys[0]}
    P["jet"] = np.concatenate(jet_of)
    labels = np.concatenate(jet_label)
    P["cls"] = labels[P["jet"]]
    P["mult"] = np.bincount(P["jet"])[P["jet"]]
    top = {i: np.concatenate(v) for i, v in rec["top"].items()}
    probs = {i: np.concatenate(v) for i, v in rec["probs"].items()}
    return P, top, probs, labels, np.concatenate(jet_scores)


def nmi(a, b):
    """Normalised mutual information I(a;b)/min(H(a),H(b)) for integer arrays."""
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    p = joint / joint.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    h = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return float(mi / max(min(h(pa), h(pb)), 1e-12))


def qbin(x, n=10):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))
    return np.searchsorted(edges, x)


def summarise(P, top, probs, n_experts):
    variables = {"particle type": P["ptype"], "charge": (P["charge"] + 1).astype(int),
                 "jet class": P["cls"]}
    for k, lab in CONT:
        variables[lab] = qbin(P[k])
    res = {}
    for i, t in top.items():
        load = np.bincount(t, minlength=n_experts) / len(t)
        ent = -(probs[i].astype(np.float32) * np.log(probs[i].astype(np.float32) + 1e-9)).sum(1).mean()
        by_type = np.zeros((len(PTYPES), n_experts))
        np.add.at(by_type, (P["ptype"], t), 1)
        by_type /= by_type.sum(1, keepdims=True).clip(min=1)
        by_cls = np.zeros((int(P["cls"].max()) + 1, n_experts))
        np.add.at(by_cls, (P["cls"], t), 1)
        by_cls /= by_cls.sum(1, keepdims=True).clip(min=1)
        profile = {k: [float(P[k][t == e].mean()) if (t == e).any() else None for e in range(n_experts)]
                   for k, _ in CONT}
        profile["charged_frac"] = [float((P["charge"][t == e] != 0).mean()) if (t == e).any() else None
                                   for e in range(n_experts)]
        res[i] = {"load": load.tolist(), "router_entropy": float(ent), "max_entropy": float(np.log(n_experts)),
                  "p_expert_given_type": by_type.tolist(), "p_expert_given_class": by_cls.tolist(),
                  "profile": profile, "nmi": {k: nmi(t, v) for k, v in variables.items()}}
    return res


@torch.no_grad()
def knockout(model, loader, device, max_jets, max_pairs, layers, n_experts):
    """Per-class accuracy with each (layer, expert) removed (router renormalises
    over the remaining experts)."""
    def per_class_acc():
        s, y, n = [], [], 0
        for batch in loader:
            feats, pairs, mask, lab, _ = prepare_batch(batch, device)
            N = feats.shape[1]
            micro = max(1, max_pairs // (N * N))
            for k in range(0, len(lab), micro):
                sl = slice(k, k + micro)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    s.append(model(feats[sl], pairs[sl], mask=mask[sl]).argmax(-1).cpu().numpy())
            y.append(lab.cpu().numpy())
            n += len(lab)
            if n >= max_jets:
                break
        s, y = np.concatenate(s)[:max_jets], np.concatenate(y)[:max_jets]
        C = int(y.max()) + 1
        return np.array([(s[y == c] == c).mean() if (y == c).any() else np.nan for c in range(C)])

    base = per_class_acc()
    delta = {}
    for i in layers:
        rows = []
        for e in range(n_experts):
            model.disable_experts({i: [e]})
            rows.append((per_class_acc() - base).tolist())
        delta[i] = rows
    model.disable_experts(None)
    return base.tolist(), delta


def plots(res, ko, class_names, n_experts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(res)
    for i in layers:
        r = res[i]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
        im = ax[0].imshow(r["p_expert_given_type"], aspect="auto", cmap="viridis", vmin=0)
        ax[0].set_yticks(range(len(PTYPES)), PTYPES); ax[0].set_xticks(range(n_experts))
        ax[0].set_xlabel("expert"); ax[0].set_title(f"layer {i}: P(expert | particle type)")
        fig.colorbar(im, ax=ax[0])
        keys = [k for k, _ in CONT] + ["charged_frac"]
        M = np.array([[np.nan if v is None else v for v in r["profile"][k]] for k in keys], dtype=float)
        Z = (M - np.nanmean(M, 1, keepdims=True)) / (np.nanstd(M, 1, keepdims=True) + 1e-9)
        im = ax[1].imshow(Z, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax[1].set_yticks(range(len(keys)), [dict(CONT).get(k, "charged frac") for k in keys])
        ax[1].set_xticks(range(n_experts), [f"{e}\n{100 * r['load'][e]:.0f}%" for e in range(n_experts)])
        ax[1].set_title("mean physics of particles routed to each expert (z-score across experts)")
        for (a, b), v in np.ndenumerate(M):
            if not np.isnan(v):
                ax[1].text(b, a, f"{v:.2g}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax[1])
        fig.tight_layout(); fig.savefig(os.path.join(out, f"routing_layer{i}.png"), dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.8), squeeze=False)
    for ax, i in zip(axs[0], layers):
        P = np.array(res[i]["p_expert_given_class"])
        ax.imshow(P / P.mean(0, keepdims=True).clip(min=1e-9), aspect="auto", cmap="RdBu_r", vmin=0, vmax=2)
        ax.set_title(f"layer {i}"); ax.set_xticks(range(n_experts))
        ax.set_yticks(range(len(P)), class_names[:len(P)])
    fig.suptitle("expert usage per jet class, relative to the average (1 = typical)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "class_usage.png"), dpi=130); plt.close(fig)

    names = list(res[layers[0]]["nmi"])
    M = np.array([[res[i]["nmi"][n] for n in names] for i in layers])
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 0.45 * len(layers) + 2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(layers)), [f"layer {i}" for i in layers])
    for (a, b), v in np.ndenumerate(M):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", color="w" if v < M.max() * 0.6 else "k", fontsize=8)
    ax.set_title("normalised mutual information: top-1 expert vs physics variable")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(out, "mutual_information.png"), dpi=130)
    plt.close(fig)

    if ko:
        base, delta = ko
        kl = sorted(delta)
        fig, axs = plt.subplots(1, len(kl), figsize=(3.2 * len(kl), 3.8), squeeze=False)
        lim = max(1e-3, max(abs(v) for i in kl for row in delta[i] for v in row if v == v))
        for ax, i in zip(axs[0], kl):
            D = np.array(delta[i]).T * 100
            ax.imshow(D, aspect="auto", cmap="RdBu", vmin=-lim * 100, vmax=lim * 100)
            ax.set_title(f"layer {i}"); ax.set_xticks(range(n_experts)); ax.set_xlabel("expert removed")
            ax.set_yticks(range(len(D)), class_names[:len(D)])
        fig.suptitle("change in per-class accuracy (percentage points) when one expert is removed")
        fig.tight_layout(); fig.savefig(os.path.join(out, "knockout.png"), dpi=130); plt.close(fig)


def describe(res, ko, class_names, n_experts):
    """Plain-language findings: dominant particle type / extreme physics per expert."""
    lines = []
    keys = dict(CONT)
    for i in sorted(res):
        r = res[i]
        lines.append(f"### Layer {i}")
        lines.append(f"router entropy {r['router_entropy']:.2f} / {r['max_entropy']:.2f} nats; "
                     f"strongest link: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                     sorted(r["nmi"].items(), key=lambda kv: -kv[1])[:3]))
        T = np.array(r["p_expert_given_type"])
        for e in range(n_experts):
            if r["load"][e] < 1e-3:
                lines.append(f"- expert {e}: unused")
                continue
            enrich = T[:, e] / max(r["load"][e], 1e-9)
            t = int(np.argmax(enrich))
            z = {k: v[e] for k, v in r["profile"].items() if v[e] is not None}
            mean = {k: np.nanmean([x for x in r["profile"][k] if x is not None]) for k in z}
            std = {k: np.nanstd([x for x in r["profile"][k] if x is not None]) + 1e-9 for k in z}
            ext = sorted(((z[k] - mean[k]) / std[k], k) for k in z)
            hi = [keys.get(k, "charged frac") for s, k in ext[::-1][:2] if s > 1]
            lo = [keys.get(k, "charged frac") for s, k in ext[:2] if s < -1]
            C = np.array(r["p_expert_given_class"])[:, e]
            fav = class_names[int(np.argmax(C / C.mean()))] if C.mean() > 0 else "-"
            s = f"- expert {e} ({100 * r['load'][e]:.0f}% of particles): {enrich[t]:.1f}x enriched in {PTYPES[t]}"
            if hi: s += "; high " + ", ".join(hi)
            if lo: s += "; low " + ", ".join(lo)
            s += f"; most used by {fav} jets"
            if ko and i in ko[1]:
                d = np.array(ko[1][i][e])
                w = int(np.nanargmin(d))
                s += f"; removing it costs {class_names[w]} {-100 * d[w]:.1f} pp"
            lines.append(s)
        lines.append("")
    return "\n".join(lines)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load(a.ckpt, device)
    if a.synthetic:
        loader = synthetic_loader(cfg, 4, 64, seed=123)
        names = [f"class_{i}" for i in range(cfg.n_classes)]
    else:
        loader = real_loader(a.cache_dir, a.val_cache, a.batch_size, False, max_jets=a.jets)
        names = CLASS_LIST
    os.makedirs(a.out, exist_ok=True)
    P, top, probs, labels, scores = collect(model, loader, device, a.jets, a.max_pairs)
    print(f"{len(labels):,} jets, {len(P['jet']):,} particles, accuracy {(scores.argmax(1) == labels).mean():.4f}")
    res = summarise(P, top, probs, cfg.n_experts)
    ko = None
    if a.knockout_jets > 0:
        kl = sorted(res) if a.knockout_layers == "all" else [int(x) for x in a.knockout_layers.split(",")]
        ko = knockout(model, loader, device, a.knockout_jets, a.max_pairs, kl, cfg.n_experts)
    plots(res, ko, names, cfg.n_experts, a.out)
    text = describe(res, ko, names, cfg.n_experts)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump({"layers": res, "knockout": {"base": ko[0], "delta": ko[1]} if ko else None,
                   "n_jets": int(len(labels)), "n_particles": int(len(P["jet"]))}, f, indent=1)
    with open(os.path.join(a.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(f"# Expert specialisation: {a.ckpt}\n\n{len(labels):,} jets, {len(P['jet']):,} particles.\n\n"
                "![MI](mutual_information.png)\n\n![class usage](class_usage.png)\n\n"
                + ("![knockout](knockout.png)\n\n" if ko else "") + text)
    print(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cache_dir", default="D:/JetClass/cache")
    ap.add_argument("--val_cache", default="val_val_50000_128")
    ap.add_argument("--jets", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_pairs", type=int, default=4_000_000)
    ap.add_argument("--knockout_jets", type=int, default=20_000, help="0 = skip knockout study")
    ap.add_argument("--knockout_layers", default="all")
    ap.add_argument("--out", default="analysis")
    main(ap.parse_args())
