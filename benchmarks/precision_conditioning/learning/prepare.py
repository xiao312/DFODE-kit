"""Execute a bounded extension of the existing reference pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REFERENCE = Path(__file__).resolve().parents[1] / "reference"


def plan():
    experiment = json.loads(Path(__file__).with_name("experiment.json").read_text())
    config = json.loads((REFERENCE / "pilot.json").read_text())
    config["temperatures_K"] = experiment["temperatures_K"]
    return config, experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/representation/run-001"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, experiment = plan()
    print(json.dumps({"intervals": 384, "solves": 3456, "split": experiment["split_by_temperature"], "output": str(args.output)}), flush=True)
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f"Run already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config_path = args.output.with_suffix(".config.json")
        with config_path.open("x") as stream:
            json.dump(config, stream, indent=2)
        subprocess.run([sys.executable, str(REFERENCE / "pilot.py"), "--config", str(config_path),
                        "--output", str(args.output)], check=True, timeout=experiment["wall_time_seconds"] + 30)
        (args.output / "experiment.json").write_text(json.dumps(experiment, indent=2))
