# eth_batched.py
import torch
from torch import nn

try:
    from functorch import vmap, grad as fgrad
    FUNCTORCH_AVAILABLE = True
except Exception:
    FUNCTORCH_AVAILABLE = False

from aqs_ultralow_latency import agape_score_ultralow_latency
from aqs_realtime import quantum_bridge_call


class EthicalScoringBatched(nn.Module):
    """
    Batched fairness: compute per-sample gradients to estimate group Fisher information.
    Uses functorch.vmap when available for speed; otherwise uses a fallback loop.
    """

    def __init__(self, lambda_aqs=0.05, fairness_weight=0.01, veto_threshold=0.78):
        super().__init__()
        self.lambda_aqs = lambda_aqs
        self.fairness_weight = fairness_weight
        self.veto_threshold = veto_threshold
        self.register_buffer("aqs_stream", torch.tensor(0.0))

    @staticmethod
    def _flatten_params(params):
        return torch.cat([p.reshape(-1) for p in params])

    def per_sample_grads(self, model, loss_fn, inputs, targets):
        """
        Returns: grads tensor shape (batch, param_count)
        Two modes:
          - functorch: efficient vmap over examples
          - fallback: loop with torch.autograd.grad (slower)
        """
        params = [p for p in model.parameters() if p.requires_grad]

        if FUNCTORCH_AVAILABLE:
            # functorch version: create a function mapping params to loss and vmap over batch
            def loss_from_params(*flat_params, x, y):
                # Helper to map flattened params -> model params is left to user (complex).
                # Simpler approach: use functional module from functorch if needed.
                raise NotImplementedError("Plug in functorch.functional API per model.")

            # NOTE: Full functorch functionalization depends on user model;
            # leaving a placeholder here because it requires converting to functional model.
            raise RuntimeError(
                "functorch path requires functionalized model (use make_functional). "
                "See fallback loop version below or convert model to functorch.functional."
            )
        else:
            # Fallback: compute per-sample grads by looping (works for any model)
            model.zero_grad()
            grads_list = []
            for i in range(inputs.size(0)):
                out = model(inputs[i:i+1])
                loss = loss_fn(out, targets[i:i+1])
                grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                flat = self._flatten_params(grads).detach().clone()  # detach to avoid storing graph
                grads_list.append(flat)
                model.zero_grad()
            grads_batch = torch.stack(grads_list, dim=0)  # (B, P)
            return grads_batch

    def compute_group_fisher(self, grads_batch, group_labels):
        """
        grads_batch: (B, P)
        group_labels: (B,) integer groups
        returns fairness_score in [0,1]
        """
        unique = torch.unique(group_labels)
        fisher_vals = []
        for g in unique:
            mask = group_labels == g
            if mask.sum() == 0:
                continue
            gp = grads_batch[mask]
            fisher_vals.append((gp * gp).mean())
        if len(fisher_vals) == 0:
            return 1.0
        fisher = torch.stack(fisher_vals)
        fairness = (fisher.mean() / (fisher.max() + 1e-12)).clamp(0.0, 1.0).item()
        return fairness

    def forward(self, model, inputs, targets, attn=None, group_labels=None, loss_fn=None):
        """
        model: the network (required to compute per-sample grads)
        inputs, targets: batch tensors
        loss_fn: loss function for per-sample grad (e.g., nn.CrossEntropyLoss(reduction='none'))
        attn: optional attention (for AQS)
        """
        if loss_fn is None:
            raise ValueError("loss_fn required for per-sample gradient extraction.")

        # 1) compute hiddens/attn as needed for AQS (assumes model returns hidden states if needed)
        hiddens = model(inputs)  # adapt if model returns tuple

        # 2) AI-side AQS (vectorized over batch if your aqs supports it)
        # here we aggregate to scalar AQS_instant (user's aqs function shape-dependent)
        aqs_ai = agape_score_ultralow_latency(hiddens, attn, None)["AQS_instant"]

        # 3) quantum purity (detached)
        with torch.no_grad():
            purity_q = quantum_bridge_call(hiddens)
        aqs_quantum = purity_q
        aqs_hybrid = aqs_ai * aqs_quantum

        # 4) per-sample gradients (batch, P)
        grads_batch = self.per_sample_grads(model, loss_fn, inputs, targets)

        # 5) fairness
        fairness_score = 1.0
        if group_labels is not None:
            fairness_score = self.compute_group_fisher(grads_batch, group_labels)

        # 6) streaming AQS update (no grad)
        with torch.no_grad():
            if self.aqs_stream == 0:
                self.aqs_stream.copy_(aqs_hybrid.detach())
            else:
                self.aqs_stream.mul_(0.95).add_(0.05 * aqs_hybrid.detach())
            veto_flag = self.aqs_stream.item() < self.veto_threshold

        # 7) ethical loss (differentiable only w.r.t aqs_ai and model params involved there)
        ethical_loss = -self.lambda_aqs * aqs_hybrid + self.fairness_weight * (1.0 - fairness_score)
        return ethical_loss, veto_flag
