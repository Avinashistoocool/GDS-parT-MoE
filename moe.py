"""
moe.py — per-particle Mixture-of-Experts for GDS-ParT.

Every particle token in a block is routed to `top_k` of `n_experts` FFN experts
(token-choice routing, as in Switch / GShard). Only the particle-level FFNs are
replaced; attention, relational gates, pairwise bias and the CLS blocks are the
same modules as in `gds_part_colab.py`.

Why per particle: the router sees one particle's representation at a time, so
its decisions can be compared directly with that particle's physics (type,
charge, pT fraction, angular distance to the jet axis, impact parameters,
local density). If the experts specialise, the routing table *is* a learned
partition of particle phase space, and knocking out an expert shows which jet
classes depend on that region.

Padding tokens are never routed and never count towards the load-balancing loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from gds_part_colab import GDSParT, GDSParTBlock, GDSParTConfig, _ffn


class GDSParTMoEConfig(GDSParTConfig):
    """GDSParTConfig plus:
        n_experts      experts per MoE layer
        top_k          experts per particle (>= 2 so the router gets a gradient
                       through the renormalised mixture weights)
        moe_layers     indices of blocks whose FFN becomes an MoE (None = all)
        aux_weight     load-balancing loss weight (Switch-style)
        z_weight       router z-loss weight (keeps router logits small)
    """

    def __init__(self, n_experts: int = 8, top_k: int = 2, moe_layers=None,
                 aux_weight: float = 1e-2, z_weight: float = 1e-3, **kw):
        super().__init__(**kw)
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.moe_layers = list(range(self.n_layers)) if moe_layers is None else list(moe_layers)
        self.aux_weight = aux_weight
        self.z_weight = z_weight

    def base_kwargs(self):
        skip = {"n_experts", "top_k", "moe_layers", "aux_weight", "z_weight"}
        return {k: v for k, v in vars(self).items() if k not in skip}


class MoEFFN(nn.Module):
    """Token-choice top-k mixture of FFN experts over real particles only."""

    def __init__(self, cfg: GDSParTMoEConfig):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.embed_dim, cfg.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList([_ffn(cfg) for _ in range(cfg.n_experts)])
        self.register_buffer("disabled", torch.zeros(cfg.n_experts, dtype=torch.bool),
                             persistent=False)
        self.record = False      # when True (eval only), keep the last routing on CPU
        self.last = None

    def forward(self, x, mask=None):
        """x: (B, N, D); mask: (B, N) bool, True = real particle.
        Returns (y, aux) where aux = (load_balance, z_loss) scalars."""
        B, N, D = x.shape
        tok = x[mask] if mask is not None else x.reshape(B * N, D)       # (T, D)
        logits = self.router(tok).float()                                # (T, E)
        if self.disabled.any():
            logits = logits.masked_fill(self.disabled, float("-inf"))
        probs = logits.softmax(-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)                    # (T, k)
        w = (top_p / top_p.sum(-1, keepdim=True).clamp(min=1e-9)).to(tok.dtype)

        out = torch.zeros_like(tok)
        for e, expert in enumerate(self.experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, expert(tok[rows]) * w[rows, slot, None])

        T = max(tok.shape[0], 1)
        frac = F.one_hot(top_i, self.n_experts).float().sum((0, 1)) / (T * self.top_k)
        live = (~self.disabled).float()
        n_live = live.sum().clamp(min=1)
        balance = n_live * (frac * probs.mean(0) * live).sum()
        finite = logits.masked_fill(self.disabled, 0.0) if self.disabled.any() else logits
        z = torch.logsumexp(finite, -1).pow(2).mean()

        if self.record and not self.training:
            self.last = {"probs": probs.detach().cpu(), "top": top_i.detach().cpu()}

        if mask is not None:
            y = x.new_zeros(B, N, D)
            y[mask] = out
        else:
            y = out.view(B, N, D)
        return y, torch.stack([balance, z])


class GDSParTMoEBlock(GDSParTBlock):
    def __init__(self, cfg: GDSParTMoEConfig, moe: bool):
        super().__init__(cfg)
        if moe:
            self.ffn = MoEFFN(cfg)

    def forward(self, x, rel, gates, attn_mask, key_padding_mask):
        h = self.norm1(x)
        if self.gate is not None:
            gate_q, gate_k = self.gate(rel, key_padding_mask, x=h)
        else:
            gate_q, gate_k = gates if gates is not None else (None, None)
        x = x + self.attn(h, gate_q, gate_k, attn_mask)
        if isinstance(self.ffn, MoEFFN):
            y, aux = self.ffn(self.norm2(x), key_padding_mask)
        else:
            y, aux = self.ffn(self.norm2(x)), x.new_zeros(2, dtype=torch.float32)
        return x + y, aux


class GDSParTMoE(GDSParT):
    """GDS-ParT whose particle FFNs are per-particle MoE layers.
    forward(...) returns logits; the summed auxiliary losses of the last forward
    are in `self.aux_loss` (already weighted, add it to the task loss)."""

    def __init__(self, cfg: GDSParTMoEConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [GDSParTMoEBlock(cfg, i in cfg.moe_layers) for i in range(cfg.n_layers)])
        self.aux_loss = torch.zeros(())

    def moe_layers(self):
        return [(i, b.ffn) for i, b in enumerate(self.blocks) if isinstance(b.ffn, MoEFFN)]

    def set_record(self, on: bool):
        for _, m in self.moe_layers():
            m.record, m.last = on, None

    def disable_experts(self, spec=None):
        """spec: {layer: [expert, ...]} to knock out; None re-enables everything."""
        for i, m in self.moe_layers():
            m.disabled.zero_()
            for e in (spec or {}).get(i, []):
                m.disabled[e] = True

    def forward(self, particle_features, pairwise_features, mask=None):
        x = self.particle_embed(particle_features)
        rel, attn_mask, gates = self._pair_inputs(pairwise_features, mask)
        ckpt = self.training and torch.is_grad_enabled()
        aux = []
        for block in self.blocks:
            if ckpt:
                x, a = torch.utils.checkpoint.checkpoint(
                    block, x, rel, gates, attn_mask, mask, use_reentrant=False)
            else:
                x, a = block(x, rel, gates, attn_mask, mask)
            aux.append(a)
        a = torch.stack(aux).sum(0)
        self.aux_loss = self.cfg.aux_weight * a[0] + self.cfg.z_weight * a[1]

        if self.cfg.n_cls_layers > 0:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            for blk in self.cls_blocks:
                cls = blk(x, cls, key_padding_mask=mask)
            pooled = self.final_norm(cls.squeeze(1))
        else:
            x = self.final_norm(x)
            if mask is not None:
                m = mask.unsqueeze(-1).to(x.dtype)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)
        return self.classifier(pooled)


def upcycle_state_dict(dense_state: dict, model: GDSParTMoE, noise: float = 0.0) -> dict:
    """Sparse upcycling: copy a trained dense GDS-ParT into the MoE model.
    Every expert starts as a copy of that block's dense FFN (plus optional
    relative Gaussian noise to break symmetry), so the untrained MoE reproduces
    the dense model exactly. The routers keep their fresh initialisation."""
    target = model.state_dict()
    out = {}
    moe_idx = {i for i, _ in model.moe_layers()}
    for k, v in dense_state.items():
        parts = k.split(".")
        if parts[0] == "blocks" and int(parts[1]) in moe_idx and parts[2] == "ffn":
            for e in range(model.cfg.n_experts):
                nk = ".".join(parts[:3] + ["experts", str(e)] + parts[3:])
                w = v.clone()
                if noise and w.is_floating_point() and w.dim() > 1:
                    w = w + noise * w.std() * torch.randn_like(w)
                out[nk] = w
        elif k in target:
            out[k] = v
    for k, v in target.items():
        out.setdefault(k, v)
    return out
