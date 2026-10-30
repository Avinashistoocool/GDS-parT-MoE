# GDS-ParT-MoE

**A per-particle Mixture-of-Experts for the Gated Dual-Stream Particle Transformer, built to see what physics each expert learns.**

In every transformer block of [GDS-ParT](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer), the
particle feed-forward network becomes 8 expert FFNs and a router. Each real jet constituent is sent to its top-2 experts.
The router looks at one particle at a time, so you can compare each routing decision with that particle's physics:
species, charge, momentum share, angle to the jet axis, impact parameter and Lund-plane kT.

- **Sparse upcycling.** Every expert starts as a copy of a trained dense GDS-ParT FFN. The untrained MoE reproduces
  the dense model exactly, and the upcycled 8-expert model keeps **84.8%** validation accuracy.
- **Physics probes.** Routing statistics, per-expert physics profiles, normalised mutual information and expert
  knockout relate each expert to particle properties and jet classes.
- **First finding.** The first-layer router splits particles by **charge and species** (NMI 0.45 / 0.43). One expert
  takes **muons and displaced tracks** and is used most by H→bb̄ jets. Deeper layers are still close to uniform.

[Technical report (PDF)](GDS-ParT-MoE_Technical_Report.pdf){ .md-button .md-button--primary }
[Source on GitHub](https://github.com/Avinashistoocool/GDS-parT-MoE){ .md-button }

## Documentation

| | |
|---|---|
| [Technical report (PDF)](GDS-ParT-MoE_Technical_Report.pdf) | Architecture, all of the mathematics (router, top-k mixture, balance and z-losses, upcycling, schedule, metrics, analysis) and the first results |
| [First results](results.md) | The 8-expert top-2 run: accuracy, mutual information per layer, layer-0 experts, knockout |

## How it works

For a particle representation `u` in a block, the router computes routing probabilities over 8 experts, keeps the
two largest, and mixes the selected experts with weights renormalised to sum to one:

```text
pi      = softmax(W_r u)                       # 8 probabilities
T       = indices of the 2 largest entries of pi
y       = sum over e in T of  pi_e / (sum over a in T of pi_a) * FFN_e(u)
```

Padding is never routed. Training adds a Switch load-balancing loss (weight 1e-2) and a router z-loss
(weight 1e-3) per MoE layer. Attention, the relational Q/K gates, the pairwise bias and the CLS blocks are the
unchanged GDS-ParT modules.

## Quick start

```bash
git clone https://github.com/Avinashistoocool/GDS-parT-MoE && cd GDS-parT-MoE
pip install -r requirements.txt          # install torch first, matching your CUDA driver
python tests/test_moe.py                 # CPU, a few seconds

# fine-tune an upcycled MoE on the JetClass cache built by the GDS-ParT trainer
python train_moe.py --cache_dir D:/JetClass/cache --train_cache train_train_10M_100000_128 \
    --val_cache val_val_50000_128 \
    --init_dense ../gds_part/checkpoints/v2_shared_10M_10ep/best.pt \
    --n_experts 8 --top_k 2 --epochs 3 --checkpoint_dir checkpoints/moe8

# what did the experts learn?
python analyze_experts.py --ckpt checkpoints/moe8/best.pt --jets 50000 --out analysis/moe8
```

!!! note
    The results so far come from a single, short fine-tune (one epoch on 300k jets, one seed). Mutual information and
    expert enrichment are correlations, not mechanisms. See the limitations section of the technical report.

## Citing

Use GitHub's **Cite this repository** button on the
[repository page](https://github.com/Avinashistoocool/GDS-parT-MoE) (generated from `CITATION.cff`).
