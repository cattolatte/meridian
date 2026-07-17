"""Device selection for training and inference (CPU / CUDA / Apple MPS).

Scale runs (ADR-0001 budgets a rented GPU) need the training path to move models and
batches onto an accelerator. Polaris' own trainers (``train_contrastive``, ``pretrain``)
already follow the model's device; this module gives a single ``resolve_device`` helper so
every Meridian driver selects the same way, and Meridian's own trainers thread the choice
through explicitly.
"""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device ``spec`` to a concrete :class:`torch.device`.

    ``"auto"`` picks CUDA if available, else Apple MPS, else CPU. An explicit spec
    (``"cuda"``, ``"mps"``, ``"cpu"``, ``"cuda:1"``, ...) is honored as given.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)
