# fed_ethics_proto_v2.py (New Logic)

def batch_ethical_score(circuit: QuantumCircuit, node: str, tomo_purity: float, petz_recovery: bool = False) -> float:
    # Node bias (proxy for S_norm, steering strength) remains the baseline
    node_biases = {"Heron": 0.20, "Forte": 0.21, "Willow": 0.19}
    base_bias = node_biases.get(node, 0.20)
    
    # Mitigation Factor is the desired coherence gain
    mitigation_factor = 8.0 # Based on 8x Petz reduction (Source 2.2)
    
    # HEOM Purity Correction: AQS rewards recovery when coherence is low.
    # If Purity is 0.35, the gain factor is 1/0.35 = ~2.85x higher.
    coherence_gain_scale = 1.0 / tomo_purity
    
    if petz_recovery:
        # Hybrid AQS Gain: Base * (Petz theoretical gain) * (Reward for low Purity)
        score = base_bias * (mitigation_factor / 100.0) * coherence_gain_scale
    else:
        score = base_bias / 8.0  # Baseline low score
        
    return float(score)
