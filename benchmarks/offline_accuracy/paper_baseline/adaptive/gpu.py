"""Read-only cooperative GPU availability; this does not reserve hardware."""
import subprocess


def inventory():
    def query(flag):
        return subprocess.check_output(["nvidia-smi",flag,"--format=csv,noheader,nounits"],text=True,timeout=15)
    occupied = {line.strip() for line in query("--query-compute-apps=gpu_uuid").splitlines() if line.strip()}
    rows = []
    for line in query("--query-gpu=index,uuid,memory.used,utilization.gpu").splitlines():
        index,uuid,memory,utilization = [v.strip() for v in line.split(",")]
        rows.append(dict(index=index,uuid=uuid,memory_MiB=int(memory),utilization=int(utilization),occupied=uuid in occupied))
    return rows


def idle(row):
    return not row["occupied"] and row["memory_MiB"] < 256 and row["utilization"] < 10
