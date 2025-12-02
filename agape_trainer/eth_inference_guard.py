# eth_inference_guard.py
import torch
from torch import nn
from aqs_ultralow_latency import agape_score_ultralow_latency
from aqs_realtime import quantum_bridge_call

class InferenceGuard(nn.Module):
    """
    Lightweight inference-only guardrail.
    - No autograd interference (torch.no_grad used)
    - Computes streaming AQS and veto boolean
    """

    def __init__(self, veto_threshold=0.78, stream_alpha=0.05):
        super().__init__()
        self.veto_threshold = veto_threshold
        self.alpha = stream_alpha
        self.register_buffer("aqs_stream", torch.tensor(0.0))

    @torch.no_grad()
    def assess(self, hiddens, attn=None, batch_mode=True):
        """
        hiddens: (B, ...) or single example
        attn: optional
        Returns: veto_flag (bool), aqs_instant (float), aqs_stream (float)
        """
        # compute AI AQS (should support batched input or aggregate)
        aqs_ai = agape_score_ultralow_latency(hiddens, attn, None)["AQS_instant"]
        # quantum purity (blocking, but assumed ultra-low-latency)
        purity_q = quantum_bridge_call(hiddens)
        aqs_hybrid = float(aqs_ai) * float(purity_q)

        # update stream without gradients
        if float(self.aqs_stream) == 0.0:
            self.aqs_stream.copy_(torch.tensor(aqs_hybrid))
        else:
            self.aqs_stream.mul_(1.0 - self.alpha).add_(self.alpha * aqs_hybrid)

        veto = float(self.aqs_stream) < self.veto_threshold
        return veto, float(aqs_hybrid), float(self.aqs_stream)
