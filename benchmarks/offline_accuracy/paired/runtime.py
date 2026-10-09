"""Fail-closed GPU settings and a complete package-version record."""
import importlib.metadata
import os
import platform
import subprocess
import torch


def configure():
    expected = {"numpy": "2.2.6", "scipy": "1.15.3", "cantera": "3.2.0"}
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    if any(packages.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Require pinned chemistry packages: {expected}")
    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise ValueError("Require tested Torch 2.8.0+cu128 and a visible GPU; CPU fallback is disabled")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(0)
    # Exercise an actual kernel; is_available alone does not prove compatibility.
    probe = torch.ones((16, 16), device="cuda")
    if (probe @ probe).sum().item() != 4096:
        raise ValueError("GPU arithmetic smoke test failed")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()
    return dict(python=platform.python_version(), packages=dict(sorted(packages.items())),
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=properties.name, device_memory_bytes=properties.total_memory,
                driver_versions=sorted(set(driver)), threads=1, learning="float32",
                reconstruction="float64", tf32=False, autocast=False, deterministic=True,
                cublas_workspace_config=":4096:8", visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                timing="Synchronized end-to-end wall time; process time is host CPU consumption only")
