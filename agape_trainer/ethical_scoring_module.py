import torch
from torch import nn

from aqs_ultralow_latency import agape_score_ultralow_latency
from aqs_realtime import quantum_bridge_call  # <100 µs quantum purity call


class EthicalScoringModule(nn.Module):
    """
    Hybrid AQS–Fairness ethical controller with streaming coherence
    and quantum veto gating.
    """

    def __init__(self, lambda_aqs=0.05, fairness_weight=0.01, veto_threshold=0.78):
        super().__init__()

        # Hyperparameters
        self.lambda_aqs = lambda_aqs
        self.fairness_weight = fairness_weight
        self.veto_threshold = veto_threshold

        # Streaming AQS buffer
        self.register_buffer("aqs_stream", torch.tensor(0.0))

    @staticmethod
    def compute_fairness_bound(grads, group_labels):
        """
        Approximate group-wise Fisher information.
        grads: (batch, params)
        group_labels: (batch,)
        Returns ∈ [0,1].
        """
        with torch.no_grad():
            unique_groups = torch.unique(group_labels)
            fisher_vals = []

            for g in unique_groups:
                g_grads = grads[group_labels == g]
                if g_grads.numel() == 0:
                    continue
                fisher_vals.append((g_grads * g_grads).mean())

            fisher_vals = torch.stack(fisher_vals)
            max_f = fisher_vals.max()
            mean_f = fisher_vals.mean()

            fairness = torch.clamp(mean_f / max_f, 0.0, 1.0)
            return fairness.item()

    def forward(self, hiddens, attn, grads, group_labels=None):
        """
        Compute composite ethical loss + veto flag.
        """
        # AI-side AQS
        aqs_ai = agape_score_ultralow_latency(
            hiddens, attn, grads
        )["AQS_instant"]  # scalar tensor

        # Quantum purity → normalized coherence
        # (Assumed already detached from PyTorch graph)
        with torch.no_grad():
            purity_q = quantum_bridge_call(hiddens)
        aqs_quantum = purity_q

        # Hybrid coherence
        aqs_hybrid = aqs_ai * aqs_quantum

        # Streaming exponential filter
        with torch.no_grad():
            if self.aqs_stream == 0:
                self.aqs_stream.copy_(aqs_hybrid.detach())
            else:
                self.aqs_stream.mul_(0.95).add_(0.05 * aqs_hybrid.detach())

        # Veto signal
        veto_flag = self.aqs_stream.item() < self.veto_threshold

        # Fairness term
        fairness_score = 1.0
        if group_labels is not None:
            fairness_score = self.compute_fairness_bound(grads, group_labels)

        # Ethical loss (differentiable)
        ethical_loss = (
            -self.lambda_aqs * aqs_hybrid +
            self.fairness_weight * (1.0 - fairness_score)
        )

        return ethical_loss, veto_flag
