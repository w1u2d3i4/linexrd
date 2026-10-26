"""Independent score integrity and calibration-trained relevance checks."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import fusion


def feature(d, k, temperature):
    d = fusion.validate_inputs(d, k)
    v = fusion.knn_vote(d, temperature)
    s = d['sim']
    return np.column_stack([
        fusion.features(d, v), s[:, -1],
        np.quantile(s, 0.25, axis=1), np.quantile(s, 0.75, axis=1),
    ])


def fit(clean, mismatched, k, temperature, seed=0):
    """Fit relevance on clean and recomputed mismatched calibration evidence."""
    x = np.concatenate([feature(clean, k, temperature),
                        feature(mismatched, k, temperature)])
    y = np.r_[np.ones(len(clean['p'])), np.zeros(len(mismatched['p']))]
    estimator = HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=20,
        early_stopping=False, random_state=seed,
    ).fit(x, y)
    return dict(estimator=estimator, K=k, temperature=temperature,
                score_tolerance=1e-4)


def apply(model, payload, candidate, threshold, integrity_error):
    """Use the same acceptance mask for any candidate; reject to PST."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Relevance threshold must be in [0, 1]")
    p = fusion.normalize(payload['p'])
    candidate = fusion.normalize(candidate)
    error = np.asarray(integrity_error, dtype=float)
    if error.shape != (len(p),) or candidate.shape != p.shape:
        raise ValueError("Evidence, probabilities, and integrity errors must align")
    confidence = model['estimator'].predict_proba(
        feature(payload, model['K'], model['temperature'])
    )[:, 1]
    accepted = ((confidence >= threshold) & np.isfinite(error)
                & (error <= model['score_tolerance']) & (error >= 0))
    return np.where(accepted[:, None], candidate, p), accepted
