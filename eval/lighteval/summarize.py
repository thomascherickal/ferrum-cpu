"""Collapse the newest LightEval results_*.json below a results dir into one summary file.

    python summarize.py <results_dir> <out.json> <model> [<max_samples>]
"""
import glob
import json
import os
import sys

LIGHTEVAL_VERSION = "0.13.0"


def summarize(res_dir, out, model, max_samples=None):
    """Write {lighteval_version, model, max_samples, tasks} to `out`; return the per-task metrics."""
    found = glob.glob(os.path.join(res_dir, "**", "results_*.json"), recursive=True)
    if not found:
        raise SystemExit(f"summarize: no results_*.json under {res_dir}")
    with open(max(found, key=os.path.getmtime)) as f:
        results = json.load(f)["results"]
    tasks = {k: v for k, v in results.items() if k != "all"}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    summary = {
        "lighteval_version": LIGHTEVAL_VERSION,
        "model": model,
        "max_samples": int(max_samples) if max_samples else None,
        "tasks": tasks,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return tasks


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    print(json.dumps(summarize(*argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
