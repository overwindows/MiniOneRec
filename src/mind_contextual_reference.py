"""Independent NumPy reference for frozen contextual scoring, not the transformer."""

import math

import numpy as np


def contextual_reference(model, histories, candidates, groups, position):
    if model.training:
        raise ValueError("Frozen evaluation is required for the contextual reference")
    candidate, group = int(candidates[position]), int(groups[position])
    history = histories[group]
    identities = np.concatenate((history, [candidate]))
    weights = {name: value.detach().numpy().astype(np.float64)
               for name, value in model.state_dict().items()}
    titles = model.features.detach().numpy()[identities].astype(np.float64)
    taxonomy = np.concatenate((
        weights["category.weight"][model.corpus.category[identities]],
        weights["subcategory.weight"][model.corpus.subcategory[identities]]), axis=1)
    encoded = (titles @ weights["projection.weight"].T
               + taxonomy @ weights["taxonomy_projection.weight"].T)
    encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
    candidate_vector = encoded[-1]
    valid = history != 0
    context = np.zeros(64, dtype=np.float64)
    if valid.any():
        logits = 8 * (encoded[:-1][valid] @ candidate_vector) + weights["recency"][valid]
        attention = np.exp(logits - logits.max())
        attention /= attention.sum()
        context = attention @ encoded[:-1][valid]
        context /= max(float(np.linalg.norm(context)), 1e-12)
    features = np.concatenate((candidate_vector, context, candidate_vector * context))
    hidden = weights["head.0.weight"] @ features + weights["head.0.bias"]
    hidden = .5 * hidden * (1 + np.asarray([math.erf(float(value) / math.sqrt(2))
                                          for value in hidden]))
    residual = float((weights["head.2.weight"] @ hidden + weights["head.2.bias"])[0])
    return (float(model.priors.pooled[candidate, 0])
            + float(weights["semantic_scale"]) * float(candidate_vector @ context) + residual)
