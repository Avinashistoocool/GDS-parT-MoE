# GDS-ParT-MoE: per-particle Mixture-of-Experts for physics interpretability

This repo adds a **per-particle Mixture-of-Experts (MoE)** to the [Gated Dual-Stream Particle Transformer](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer). It then measures **what physics each expert learns** on JetClass.

## Idea

In every transformer block, the particle FFN becomes `n_experts` FFNs and a router. The router sends each **particle token** to its top-k experts. Because the router looks at one particle at a time, you can compare each routing decision with that particle's physics. If the experts specialise, the routing works as a learned partition of particle phase space:

| question | measured by |
|---|---|
| Does an expert handle one particle species (photons, neutral hadrons, leptons)? | P(expert \| particle type) |
| Does it handle the hard core or the soft, wide-angle radiation? | mean ln pT, pT/pT_jet, ΔR to axis, pT rank per expert |
| Does it handle displaced tracks (b/c decays)? | mean \|tanh d0\| per expert |
| Does it handle collinear splittings (Lund plane)? | ΔR and ln kT to the nearest neighbour |
| Which variable explains the routing best? | normalised mutual information MI(expert; variable) per layer |
| Which jet classes depend on this expert? | knockout: Δ per-class accuracy with the expert removed |

Attention, the relational Q/K gates, the pairwise bias and the CLS blocks are the original `gds_part_colab.py` modules, unchanged.

## Sparse upcycling

`--init_dense` copies a trained dense GDS-ParT checkpoint into the MoE, with every expert starting as that block's FFN. Before any training, the MoE reproduces the dense model exactly; `tests/test_moe.py` checks this. A short fine-tune then lets the experts diverge. On the v2 10M checkpoint, the upcycled model reached val acc **0.8479** before fine-tuning (dense: 0.8487).

## Files

- `moe.py` contains `MoEFFN`, a token-choice top-k router over real particles only. Padding is never routed. It also has the Switch load-balance and z-losses, routing recording, and expert knockout. The file also defines `GDSParTMoE` and `upcycle_state_dict`.
- `train_moe.py` trains the model on the JetClass memmap cache from the GDS-ParT repo. It uses a separate router learning rate and AMP, and splits large batches into micro-batches.
- `analyze_experts.py` covers routing statistics, physics profiles, mutual information, knockout plots, and `REPORT.md`.
- `metrics.py` contains the ParT-paper jet-tagging metrics.
- `gds_part_colab.py` and `data/jetclass_cache.py` are copied from the GDS-ParT repo.

## Usage

```bash
pip install -r requirements.txt
python tests/test_moe.py                                     # CPU, a few seconds

# fine-tune an upcycled MoE (cache built by the GDS-ParT trainer)
python train_moe.py --cache_dir D:/JetClass/cache --train_cache train_train_10M_100000_128 \
    --val_cache val_val_50000_128 \
    --init_dense ../gds_part/checkpoints/v2_shared_10M_10ep/best.pt \
    --n_experts 8 --top_k 2 --epochs 3 --checkpoint_dir checkpoints/moe8

# what did the experts learn?
python analyze_experts.py --ckpt checkpoints/moe8/best.pt --jets 50000 --out analysis/moe8
```

Useful knobs: `--moe_layers 3,5,7` puts MoE only in the deeper blocks, which are usually more specialised. `--aux_weight` trades load balance against specialisation; lower it if every expert looks the same. `--top_k 1` gives harder, easier-to-read routing.

## First result (preliminary)

The v2 checkpoint was upcycled to 8 experts with top-2 routing in all 8 layers. It was fine-tuned for 1 epoch on 300k jets and analysed on 20k validation jets (782k particles). Full report: [`analysis/moe8_short/REPORT.md`](analysis/moe8_short/REPORT.md).

- Validation accuracy was 0.8472 (dense model: 0.8487).
- **Layer 0 already partitions by species.** The NMI between the chosen expert and charge is 0.45, and with particle type it is 0.43. The experts split roughly into:
  - muons and displaced tracks (4.5× muon, high |d0|, Hbb-heavy)
  - photons, both soft wide-angle and collinear
  - neutral hadrons
  - hard leading charged hadrons
- **Deeper layers are still almost uniform.** Router entropy is about 2.07 of a 2.08 maximum, and knockouts change accuracy by at most 0.2 pp. The upcycled experts have not diverged yet. Train longer with the 10M cache and a lower `--aux_weight`, or use `--top_k 1`, before drawing conclusions about deeper-layer physics.

## Reading the results

- Check **NMI near 0 in early layers and rising with depth**. That pattern means routing moves from raw kinematics to jet-substructure roles.
- A **knockout that hurts mainly Hbb/Hcc** for an expert with high |d0| is a displaced-vertex (heavy-flavour) expert. One that hurts **Tbl/Hqql** and is enriched in e/μ is a lepton expert.
- These statements are correlations. Confirm a finding by retraining with a different seed and checking that the same partition appears.
