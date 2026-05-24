"""
Adaptive Contrastive Logit Subtraction (ACLS) and ACLS-Lite mitigation.
"""

import torch
import torch.nn.functional as F

def adaptive_contrastive_logit_subtraction(model, processor, image, query, perturbed_image,
                                          H_ms, alpha=0.5, tau=2.1, return_logits=False):
    """
    Perform ACLS at inference time. Requires model that can output logits for tokens.
    For API models without logit access, use prompt-level adaptation (not implemented here).
    """
    # Get original logits
    inputs = processor(text=query, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
        logits_orig = outputs.logits[0, -1, :]  # next token logits

    # Get perturbed image logits
    inputs_pert = processor(text=query, images=perturbed_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_pert = model(**inputs_pert, output_hidden_states=False)
        logits_pert = outputs_pert.logits[0, -1, :]

    # Compute hallucination probability
    P_hall = torch.sigmoid(torch.tensor(H_ms - tau))
    # Adjusted logits
    logits_adj = logits_orig - alpha * P_hall * logits_pert

    if return_logits:
        return logits_adj
    else:
        # Generate token from adjusted logits
        probs = F.softmax(logits_adj, dim=-1)
        next_token = torch.argmax(probs)
        return next_token

def acls_lite(model, processor, image, query, perturbed_image, alpha_model):
    """
    ACLS-Lite: fixed alpha per model, no entropy computation.
    """
    inputs = processor(text=query, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_orig = model(**inputs).logits[0, -1, :]
    inputs_pert = processor(text=query, images=perturbed_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_pert = model(**inputs_pert).logits[0, -1, :]
    logits_adj = logits_orig - alpha_model * logits_pert
    probs = F.softmax(logits_adj, dim=-1)
    return torch.argmax(probs)
