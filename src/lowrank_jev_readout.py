"""Compact whitened residuals that preserve the retained rich text margin."""

import numpy as np

from streamed_jev_readout import BATCH_SIZE, check_deadline, predict_streamed


def gaussian_projection(dimension, components=64):
    if not 1 <= components <= dimension:
        raise ValueError("Projection components must be positive and no larger than input dimension")
    return np.random.default_rng(42).normal(size=(dimension, components)) / np.sqrt(components)


def project(features, projection, deadline=None):
    if features.ndim != 2 or projection.ndim != 2 or features.shape[1] != projection.shape[0]:
        raise ValueError("Projection and input dimensions differ")
    if not np.isfinite(projection).all():
        raise ValueError("Projection must be finite")
    values = np.empty((len(features), projection.shape[1]), dtype=np.float64)
    for start in range(0, len(features), BATCH_SIZE):
        check_deadline(deadline)
        end = start + BATCH_SIZE
        values[start:end] = np.asarray(features[start:end], dtype=np.float64) @ projection
    if not np.isfinite(values).all():
        raise ValueError("Projected features must be finite")
    return values


def fit_whitening(values, fit_mask):
    fit_mask = np.asarray(fit_mask)
    if (values.ndim != 2 or fit_mask.shape != (len(values),)
            or fit_mask.dtype != bool or fit_mask.sum() < 2):
        raise ValueError("Whitening needs at least two aligned fitting rows")
    fitting = values[fit_mask]
    if not np.isfinite(fitting).all():
        raise ValueError("Whitening fitting values must be finite")
    center = fitting.mean(axis=0)
    fitting -= center
    covariance = fitting.T @ fitting / len(fitting)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    retained = eigenvalues > max(float(eigenvalues[-1]) * 1e-6, 1e-12)
    if not retained.any():
        raise ValueError("Whitening cannot fit entirely constant features")
    whitening = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
    return {"center": center, "whitening": whitening}


def transform(values, encoding):
    center, whitening = encoding["center"], encoding["whitening"]
    if (values.ndim != 2 or center.shape != (values.shape[1],)
            or whitening.ndim != 2 or whitening.shape[0] != values.shape[1]
            or not whitening.shape[1] or not np.isfinite(center).all()
            or not np.isfinite(whitening).all()):
        raise ValueError("Whitening dimensions or parameters differ")
    output = (values - center) @ whitening
    if not np.isfinite(output).all():
        raise ValueError("Whitened features must be finite")
    return output


def predict_lowrank(encoding, head, features, extras, retained_text, deadline=None):
    if extras.ndim != 2 or len(extras) != len(features):
        raise ValueError("Low-rank popularity features must align with frozen features")
    projected = project(features, encoding["projection"], deadline)
    compact = transform(np.column_stack((projected, extras)), encoding)
    return predict_streamed(head, compact, np.empty((len(compact), 0)), retained_text)


def save_model(path, encoding, head):
    np.savez(
        path, projection=encoding["projection"], center=encoding["center"],
        whitening=encoding["whitening"],
        **{"head_" + key: value for key, value in head.items()},
    )
