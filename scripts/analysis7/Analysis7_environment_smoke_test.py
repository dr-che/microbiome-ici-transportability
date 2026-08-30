#!/usr/bin/env python
"""Compatibility and determinism smoke test using only synthetic data."""
from __future__ import annotations
import json
import platform
import random
from pathlib import Path

import numpy as np

SEED = 20260802

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    import pytorch_lightning as pl
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def run_once(seed: int):
    set_seed(seed)
    import pandas as pd
    import sklearn
    import torch
    import pytorch_lightning as pl
    import debiasm
    from debiasm import DebiasMClassifier, OnlineDebiasMClassifier

    rng = np.random.default_rng(seed)
    n_batches = 4
    n_per_batch = 16
    n_features = 20
    batches = np.repeat(np.arange(n_batches), n_per_batch)
    X = rng.gamma(shape=1.0, scale=1.0, size=(len(batches), n_features))
    X = X / X.sum(axis=1, keepdims=True)
    y = rng.integers(0, 2, size=len(batches))
    Xb = np.column_stack([batches, X])

    target = batches == (n_batches - 1)
    X_train, X_target = Xb[~target], Xb[target]
    y_train = y[~target]

    standard = DebiasMClassifier(x_val=X_target, random_state=seed, min_epochs=25)
    standard.fit(X_train, y_train)
    z_standard = np.asarray(standard.transform(X_target), dtype=float)
    p_standard = np.asarray(standard.predict_proba(X_target), dtype=float)[:, 1]

    online = OnlineDebiasMClassifier(random_state=seed, min_epochs=25)
    online.fit(X_train, y_train)
    z_online = np.asarray(online.transform(X_target), dtype=float)
    # Avoid a second 10,000-iteration online adaptation.
    with torch.no_grad():
        p_online = torch.softmax(
            online.model.linear(torch.tensor(z_online).float()), dim=1
        )[:, 1].cpu().numpy()

    for name, p, z in [
        ("standard", p_standard, z_standard),
        ("online", p_online, z_online),
    ]:
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise RuntimeError(f"{name}: invalid probabilities")
        if not np.isfinite(z).all():
            raise RuntimeError(f"{name}: non-finite transformed abundance")
        if not np.allclose(z.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"{name}: transformed rows do not sum to one")

    return {
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
            "debiasm": getattr(debiasm, "__version__", "0.0.2"),
        },
        "standard_probabilities": p_standard.tolist(),
        "online_probabilities": p_online.tolist(),
    }

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    out.mkdir(exist_ok=True)

    first = run_once(SEED)
    second = run_once(SEED)

    a = np.asarray(first["standard_probabilities"])
    b = np.asarray(second["standard_probabilities"])
    c = np.asarray(first["online_probabilities"])
    d = np.asarray(second["online_probabilities"])

    std_diff = float(np.max(np.abs(a - b)))
    online_diff = float(np.max(np.abs(c - d)))
    passed = std_diff <= 1e-7 and online_diff <= 1e-7

    result = {
        "status": "PASS" if passed else "FAIL_NONDETERMINISTIC",
        "seed": SEED,
        "versions": first["versions"],
        "standard_max_abs_repeat_difference": std_diff,
        "online_max_abs_repeat_difference": online_diff,
    }
    (out / "smoke_test_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
