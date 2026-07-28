from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


SHUFFLE_ALGORITHM = "numpy-pcg64-seedsequence-v1"


class DeterministicEpochSampler(torch.utils.data.Sampler[int]):
    """Produce a reproducible, independently seeded permutation per epoch."""

    def __init__(self, data_source, *, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        indices = rng.permutation(len(self.data_source)).tolist()
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


def configure_reproducibility(*, seed: int, deterministic: bool) -> dict:
    """Configure process RNGs and deterministic Torch execution."""

    seed = int(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = False

    return {
        "seed": seed,
        "deterministic_algorithms": bool(deterministic),
        "shuffle_algorithm": SHUFFLE_ALGORITHM,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32_enabled": False,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash model parameters and buffers in stable state-dict order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
