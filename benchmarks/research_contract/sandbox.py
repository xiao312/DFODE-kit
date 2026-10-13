"""Allowlisted Linux candidate mounts; controller/reference files are not mounted."""
from pathlib import Path
import subprocess


def command(project, input_dir, output_dir, arguments, *, model_dir=None, gpu=0):
    project, input_dir, output_dir = map(lambda p: Path(p).resolve(), (project, input_dir, output_dir))
    if not input_dir.is_dir() or not output_dir.is_dir() or input_dir == output_dir:
        raise ValueError("Require distinct existing input/output directories")
    environment = project / "runs/envs/paired-gpu-20261009"
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
           "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache", "--proc", "/proc", "--dev", "/dev",
           "--tmpfs", "/tmp", "--ro-bind", "/sys", "/sys",
           "--ro-bind", str(environment), "/env",
           "--ro-bind", str(project / "benchmarks"), "/code/benchmarks",
           "--ro-bind", str(project / "dfode_kit"), "/code/dfode_kit",
           "--ro-bind", str(input_dir), "/input", "--bind", str(output_dir), "/output"]
    if model_dir is not None:
        cmd += ["--ro-bind", str(Path(model_dir).resolve()), "/model"]
    for device in (f"/dev/nvidia{int(gpu)}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
        if Path(device).exists():
            cmd += ["--dev-bind", device, device]
    env = dict(PATH="/env/bin:/usr/bin:/bin", HOME="/tmp", PYTHONPATH="/code", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    for name, value in env.items():
        cmd += ["--setenv", name, value]
    return cmd + ["--chdir", "/code", "/env/bin/python", *arguments]


def probe(project, input_dir, output_dir, gpu=0):
    code = """import json, os, pathlib, socket, torch
assert not pathlib.Path('/home').exists()
assert not pathlib.Path('/data2').exists()
assert not pathlib.Path('/input/training.npz').exists()
assert not pathlib.Path('/input/development.npz').exists()
assert not any(k in os.environ for k in ('HTTP_PROXY','HTTPS_PROXY','SSH_AUTH_SOCK'))
try:
    pathlib.Path('/code/benchmarks/probe-forbidden').write_text('x')
    raise AssertionError('source writable')
except OSError:
    pass
try:
    pathlib.Path('/input/probe-forbidden').write_text('x')
    raise AssertionError('inputs writable')
except OSError:
    pass
assert torch.cuda.is_available()
assert (torch.ones(4,device='cuda').sum().item()) == 4
assert socket.if_nameindex() == [(1,'lo')]
print(json.dumps({'label_paths_absent':True,'readonly_mounts':True,'network_isolated':True,'gpu':torch.cuda.get_device_name(0)}))
"""
    result = subprocess.run(command(project, input_dir, output_dir, ["-c", code], gpu=gpu),
                            env={"PATH":"/usr/bin:/bin"}, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Isolation probe failed: " + result.stderr)
    return result.stdout
