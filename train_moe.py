"""
train_moe.py — train GDS-ParT-MoE on the JetClass cache.

Typical use: upcycle the trained dense v2 checkpoint so the MoE starts at the
dense model's accuracy, then fine-tune so the experts specialise.

    python train_moe.py --cache_dir D:/JetClass/cache \
        --train_cache train_train_10M_100000_128 --val_cache val_val_50000_128 \
        --init_dense ../gds_part/checkpoints/v2_shared_10M_10ep/best.pt \
        --n_experts 8 --top_k 2 --epochs 3 --lr 2e-4 --checkpoint_dir checkpoints/moe8

    python train_moe.py --synthetic --epochs 1      # no-data dry run
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from gds_part_colab import make_synthetic_batch
from metrics import compute_jet_tagging_metrics
from moe import GDSParTMoE, GDSParTMoEConfig, upcycle_state_dict

CLASS_LIST = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def prepare_batch(batch, device):
    feats, second, mask, labels = (t.to(device, non_blocking=True) for t in batch)
    vectors = None
    if second.dim() == 3:  # (B, N, 4) four-vectors -> (B, N, N, 4) pairwise
        from data.jetclass_cache import pairwise_features
        vectors, second = second, pairwise_features(second, mask)
    return feats, second, mask, labels, vectors


def synthetic_loader(cfg, n_batches, batch_size, seed=0):
    torch.manual_seed(seed)
    batches = [make_synthetic_batch(batch_size, cfg.max_particles, cfg) for _ in range(n_batches)]
    return batches


def real_loader(cache_root, name, batch_size, shuffle, max_jets=None, seed=0):
    from data.jetclass_cache import CachedJetLoader
    return CachedJetLoader(os.path.join(cache_root, name), batch_size, shuffle=shuffle,
                           max_jets=max_jets, seed=seed)


def forward_chunked(model, feats, pairs, mask, max_pairs, use_amp):
    """Logits for a batch, split so B*N*N stays under max_pairs."""
    N = feats.shape[1]
    micro = max(1, max_pairs // (N * N))
    out = []
    for s in range(0, feats.shape[0], micro):
        sl = slice(s, s + micro)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out.append(model(feats[sl], pairs[sl], mask=mask[sl]).float())
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_pairs, max_batches=None):
    model.eval()
    scores, labels = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        feats, pairs, mask, y, _ = prepare_batch(batch, device)
        scores.append(forward_chunked(model, feats, pairs, mask, max_pairs, use_amp).softmax(-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    model.train()
    s, y = np.concatenate(scores), np.concatenate(labels)
    names = CLASS_LIST if s.shape[1] == len(CLASS_LIST) else [f"class_{i}" for i in range(s.shape[1])]
    m = compute_jet_tagging_metrics(s, y, names)
    m["per_class_accuracy"] = {n: float((s[y == c].argmax(1) == c).mean()) for c, n in enumerate(names)
                               if (y == c).any()}
    return m


def build_model(args):
    base = {}
    if args.init_dense:
        ck = torch.load(args.init_dense, map_location="cpu", weights_only=False)
        base = dict(ck["config"])
        # v1 checkpoints predate the v2 switches
        base.setdefault("gate_mode", "sigmoid"); base.setdefault("per_layer_gate", False)
        base.setdefault("pair_bias", False); base.setdefault("n_cls_layers", 0)
    else:
        base = dict(embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
                    ffn_dim=args.ffn_dim, pairwise_embed_dim=args.pairwise_embed_dim,
                    per_layer_gate=bool(args.per_layer_gate), n_cls_layers=args.n_cls_layers)
    moe_layers = None if args.moe_layers == "all" else [int(i) for i in args.moe_layers.split(",")]
    cfg = GDSParTMoEConfig(n_experts=args.n_experts, top_k=args.top_k, moe_layers=moe_layers,
                           aux_weight=args.aux_weight, z_weight=args.z_weight, **base)
    model = GDSParTMoE(cfg)
    if args.init_dense:
        model.load_state_dict(upcycle_state_dict(ck["model_state"], model, args.upcycle_noise))
        print(f"upcycled dense checkpoint {args.init_dense} "
              f"(dense val_acc={ck.get('best_val_acc') or ck.get('val_acc')})")
    return model, cfg


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp) and device == "cuda"
    if device == "cuda" and args.gpu_mem_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    model, cfg = build_model(args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    active = n_params - sum(p.numel() for _, m in model.moe_layers()
                            for e in m.experts[cfg.top_k:] for p in e.parameters())
    print(f"device {device}  params {n_params:,}  active/particle ~{active:,}  "
          f"experts {cfg.n_experts} top-{cfg.top_k} on layers {cfg.moe_layers}")

    if args.synthetic:
        train_ld = synthetic_loader(cfg, args.synthetic_batches, args.batch_size, args.seed)
        val_ld = synthetic_loader(cfg, max(2, args.synthetic_batches // 4), args.batch_size, args.seed + 1)
    else:
        train_ld = real_loader(args.cache_dir, args.train_cache, args.batch_size, True, seed=args.seed)
        val_ld = real_loader(args.cache_dir, args.val_cache, args.eval_batch_size, False,
                             max_jets=args.val_max_jets)

    # routers learn faster than the upcycled weights they route between
    router_p = [p for n, p in model.named_parameters() if ".router." in n]
    other_p = [p for n, p in model.named_parameters() if ".router." not in n]
    opt = torch.optim.AdamW([{"params": other_p}, {"params": router_p, "lr": args.lr * args.router_lr_mult}],
                            lr=args.lr, weight_decay=args.weight_decay)
    total = len(train_ld) * args.epochs
    warm = int(args.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save(name, epoch, m):
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": vars(cfg),
                    "val_acc": m["accuracy"] if m else None, "args": vars(args)},
                   os.path.join(args.checkpoint_dir, name))

    history, best, step = [], -1.0, 0
    if args.eval_init and not args.synthetic:
        m0 = evaluate(model, val_ld, device, use_amp, 2 * args.max_pairs_per_step, args.val_max_batches)
        print(f"[init] val_acc={m0['accuracy']:.4f}  auc={m0['auc_macro_ovo']:.4f}")
        history.append({"epoch": 0, "val_acc": m0["accuracy"], "val_auc": m0["auc_macro_ovo"]})
    for epoch in range(args.epochs):
        model.train()
        t0 = tl = time.time()
        run_loss = run_aux = run_correct = run_n = 0.0
        for batch in train_ld:
            feats, pairs, mask, y, _ = prepare_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            B, N = feats.shape[:2]
            micro = max(1, min(B, args.max_pairs_per_step // (N * N)))
            for s in range(0, B, micro):
                sl = slice(s, s + micro)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lg = model(feats[sl], pairs[sl], mask=mask[sl])
                    ce = F.cross_entropy(lg.float(), y[sl], label_smoothing=args.label_smoothing)
                    frac = lg.shape[0] / B
                    loss = (ce + model.aux_loss) * frac
                scaler.scale(loss).backward()
                run_loss += ce.item() * lg.shape[0]
                run_aux += model.aux_loss.item() * lg.shape[0]
                run_correct += (lg.argmax(-1) == y[sl]).sum().item()
                run_n += lg.shape[0]
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  ep {epoch + 1} step {step}/{total}  ce={run_loss / run_n:.4f}  "
                      f"aux={run_aux / run_n:.4f}  acc={run_correct / run_n:.4f}  "
                      f"{args.log_every * args.batch_size / (time.time() - tl):.0f} jets/s", flush=True)
                tl = time.time()
        m = evaluate(model, val_ld, device, use_amp, 2 * args.max_pairs_per_step, args.val_max_batches)
        print(f"[epoch {epoch + 1}] train_ce={run_loss / max(run_n, 1):.4f}  val_acc={m['accuracy']:.4f}  "
              f"val_auc={m['auc_macro_ovo']:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        history.append({"epoch": epoch + 1, "train_ce": run_loss / max(run_n, 1),
                        "val_acc": m["accuracy"], "val_auc": m["auc_macro_ovo"]})
        if m["accuracy"] > best:
            best = m["accuracy"]
            save("best.pt", epoch, m)
        save("last.pt", epoch, m)
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump({"n_params": n_params, "history": history, "best_val_acc": best}, f, indent=2)
    print(f"done. best val_acc={best:.4f}")


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic_batches", type=int, default=8)
    ap.add_argument("--cache_dir", default="D:/JetClass/cache")
    ap.add_argument("--train_cache", default="train_train_10M_100000_128")
    ap.add_argument("--val_cache", default="val_val_50000_128")
    ap.add_argument("--val_max_jets", type=int, default=100_000)
    ap.add_argument("--val_max_batches", type=int, default=None)
    ap.add_argument("--init_dense", default=None, help="dense GDS-ParT checkpoint to upcycle")
    ap.add_argument("--upcycle_noise", type=float, default=0.01,
                    help="relative noise added to each expert copy to break symmetry")
    ap.add_argument("--eval_init", type=int, default=1, help="evaluate before training (1/0)")

    ap.add_argument("--n_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--moe_layers", default="all", help='"all" or e.g. "3,5,7"')
    ap.add_argument("--aux_weight", type=float, default=1e-2)
    ap.add_argument("--z_weight", type=float, default=1e-3)
    ap.add_argument("--router_lr_mult", type=float, default=5.0)

    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=512)
    ap.add_argument("--pairwise_embed_dim", type=int, default=64)
    ap.add_argument("--per_layer_gate", type=int, default=0)
    ap.add_argument("--n_cls_layers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--max_pairs_per_step", type=int, default=2_000_000)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--gpu_mem_fraction", type=float, default=0.8)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint_dir", default="checkpoints/moe")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
