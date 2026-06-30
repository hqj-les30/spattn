"""
Analytical FLOPs / communication-cost utilities for the FLOW framework.

Conventions
-----------
- 1 MAC = 1 multiply-accumulate. Reported FLOPs = 2 * MACs (1 multiply + 1 add).
- Forward MACs are counted via forward hooks on leaf modules (no third-party deps).
- Backward pass is approximated as 2x the forward MACs (standard rule of thumb).
- Communication counts only model-parameter bytes (hyperparameters ignored), uplink only.

Used to populate the ``efficiency`` field of ``details.jsonl`` (see servers/base.py,
servers/flow.py).
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

BYTES_PER_F32 = 4


def full_param_numel(state_dict: Dict[str, torch.Tensor]) -> int:
    """Total float parameters in a full state_dict (excludes BatchNorm ``num_batches_tracked``)."""
    total = 0
    for k, v in state_dict.items():
        if k.endswith('num_batches_tracked'):
            continue
        total += v.numel()
    return total


# ---------------------------------------------------------------------------
# per-layer MAC counters. Hook signature (with with_kwargs=True):
#   (module, args: tuple, kwargs: dict, output) -> macs
# MultiheadAttention is called with keyword args in this codebase, so we must
# read kwargs (positional-only ``args`` would be empty there).
# ---------------------------------------------------------------------------

def _arg(args, kwargs, pos, name):
    """Fetch a positional-or-keyword argument (kwargs take priority by name)."""
    if name in kwargs:
        return kwargs[name]
    if pos < len(args):
        return args[pos]
    return None


def _linear_macs(module, args, kwargs, output):
    x = _arg(args, kwargs, 0, 'input')
    leading = x.numel() // x.shape[-1]
    return module.in_features * module.out_features * leading


def _conv2d_macs(module, args, kwargs, output):
    kernel = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    return kernel * output.numel()


def _norm_macs(module, args, kwargs, output):
    # normalize (mean/var) + affine: a few ops per element, approximate as numel
    return output.numel()


def _embedding_macs(module, args, kwargs, output):
    return 0  # table lookup, negligible


def _mha_macs(module, args, kwargs, output):
    """MultiheadAttention MACs: in/out projections (4 linears) + attention matmuls.

    Handles unequal query/key lengths (cross-attention) and kwargs-only calls.
    """
    q = _arg(args, kwargs, 0, 'query')
    k = _arg(args, kwargs, 1, 'key')
    if q is None:
        return 0
    if k is None:
        k = q
    B, Lq = q.shape[0], q.shape[1]
    Lk = k.shape[1]
    E = module.embed_dim
    proj = B * E * E * (2 * Lq + 2 * Lk)   # q,k,v in-proj + out-proj
    attn = 2 * B * Lq * Lk * E             # QK^T + softmax @ V
    return proj + attn


_HOOKS: Tuple = (
    (nn.Linear, _linear_macs),
    (nn.Conv1d, _conv2d_macs),
    (nn.Conv2d, _conv2d_macs),
    (nn.Conv3d, _conv2d_macs),
    (nn.BatchNorm1d, _norm_macs),
    (nn.BatchNorm2d, _norm_macs),
    (nn.BatchNorm3d, _norm_macs),
    (nn.LayerNorm, _norm_macs),
    (nn.GroupNorm, _norm_macs),
    (nn.Embedding, _embedding_macs),
    (nn.MultiheadAttention, _mha_macs),
)


def count_forward_macs(model: nn.Module, sample_inputs, sample_kwargs: Dict = None) -> int:
    """Count forward MACs for a single forward pass.

    sample_inputs : tuple of positional args passed to ``model(*sample_inputs)``.
    sample_kwargs : optional dict of keyword args.
    """
    handles = []
    bag = {'macs': 0}

    def make_hook(fn):
        # with_kwargs=True so we see keyword args (MultiheadAttention uses them)
        def hook(module, args, kwargs, output):
            try:
                bag['macs'] += fn(module, args, kwargs, output)
            except Exception:
                pass
        return hook

    for m in model.modules():
        fn = None
        for cls, f in _HOOKS:
            if isinstance(m, cls):
                fn = f
                break
        if fn is not None:
            handles.append(m.register_forward_hook(make_hook(fn), with_kwargs=True))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if sample_kwargs:
                model(*sample_inputs, **sample_kwargs)
            else:
                model(*sample_inputs)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return int(bag['macs'])


def macs_to_flops(macs: int) -> int:
    """FLOPs = 2 * MACs."""
    return 2 * int(macs)


# ---------------------------------------------------------------------------
# composed cost helpers
# ---------------------------------------------------------------------------

def client_training_flops(cnn_forward_macs: int, num_batches: int, local_epochs: int) -> int:
    """Per-client per-round training FLOPs.

    Each batch does 1 forward + 1 backward (≈2x forward) → 3x forward MACs per batch,
    converted to FLOPs (x2), across all batches and local epochs.
    """
    per_batch_macs = cnn_forward_macs * (1 + 2)
    return macs_to_flops(per_batch_macs * num_batches * local_epochs)


def forward_macs_for_model(modelfunc, data_shape: Tuple[int, ...], n_classes: int = None,
                           device: str = 'cpu') -> int:
    """Build a fresh model and count one-sample forward MACs.

    Falls back to a 2x-param-numel heuristic if the forward fails (e.g. non-float inputs).
    """
    model = modelfunc(n_class=n_classes) if n_classes is not None else modelfunc()
    model = model.to(device)
    try:
        x = torch.zeros((1,) + tuple(data_shape), device=device)
        return count_forward_macs(model, (x,))
    except Exception:
        return 2 * full_param_numel(model.state_dict())


def uplink_projected_bytes(cluster_size: int, feature_dim: int) -> int:
    """Uplink bytes for RL selection: each cluster client uploads its projected vector."""
    return cluster_size * feature_dim * BYTES_PER_F32


def uplink_full_bytes(selection_size: int, full_param_size: int) -> int:
    """Uplink bytes for aggregation: only selected clients upload full models."""
    return selection_size * full_param_size * BYTES_PER_F32
