"""Exercise the public site publisher through its actual command-line boundary."""
import json
from pathlib import Path
import subprocess
import sys


BENCHMARK = Path(__file__).parents[1] / "benchmarks/precision_conditioning"


def run_script(name, *arguments):
    return subprocess.run(
        [sys.executable, str(BENCHMARK / name), *map(str, arguments)],
        capture_output=True, text=True, check=False,
    )


def test_pages_requires_source_and_figure(tmp_path):
    result = run_script("publish_pages.py", tmp_path / "missing.json", "--output", tmp_path / "site")
    assert result.returncode != 0
    assert "must exist" in result.stderr
    assert not (tmp_path / "site").exists()


def test_pages_dry_run_and_public_artifacts(tmp_path):
    source = tmp_path / "audit.json"
    assert run_script("audit.py", "--output", source).returncode == 0
    assert run_script("plot.py", source).returncode == 0
    destination = tmp_path / "site"
    preview = run_script("publish_pages.py", source, "--output", destination, "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert not destination.exists()
    built = run_script("publish_pages.py", source, "--output", destination)
    assert built.returncode == 0, built.stderr
    files = {str(path.relative_to(destination)).replace("\\", "/")
             for path in destination.rglob("*") if path.is_file()}
    assert files == {"index.html", ".nojekyll", "publication.json",
                     "checkpoint-01/audit.json", "checkpoint-01/report.html",
                     "checkpoint-01/coordinate-audit.png"}
    landing = (destination / "index.html").read_text(encoding="utf-8")
    assert "https://github.com/users/xiao312/projects/1" in landing
    assert "No chemistry solver or model training" in landing
    assert (destination / "checkpoint-01/audit.json").read_bytes() == source.read_bytes()
    assert json.loads((destination / "publication.json").read_text())["benchmark"] == "precision_conditioning"


def test_pages_reference_allowlist(tmp_path):
    source = tmp_path / "audit.json"
    assert run_script("audit.py", "--output", source).returncode == 0
    assert run_script("plot.py", source).returncode == 0
    reference = tmp_path / "reference"
    reference.mkdir()
    allowed = {"report.html", "summary.json", "analysis-provenance.json", "reference-feasibility.png", "reference-cancellation.png"}
    for name in allowed:
        (reference / name).write_text("placeholder")
    (reference / "summary.json").write_text(json.dumps({"interval_count": 96, "assessed_species_components": 3024,
                                                      "budget_fit_count": 3012, "relative_fit_count": 2132}))
    (reference / "intervals.jsonl").write_text("private raw data")
    destination = tmp_path / "site"
    result = run_script("publish_pages.py", source, "--output", destination, "--reference-run", reference)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in (destination / "checkpoint-02").iterdir()} == allowed
    assert "Checkpoint 02 · ready" in (destination / "index.html").read_text(encoding="utf-8")
    assert (destination / "checkpoint-01/report.html").is_file()
