"""Generate submission-facing coverage and statistic audits from committed outputs."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    base = root()
    data = base / "data"
    out = base.parent / "supplementary"
    out.mkdir(parents=True, exist_ok=True)

    families = data / "families_arbert.jsonl"
    total = retained = 0
    excluded = Counter()
    condition_counts = Counter()
    alignment_failures = []
    for line in families.open(encoding="utf-8"):
        item = json.loads(line)
        total += 1
        if "arbert" not in item.get("models", {}):
            excluded[item["id"]] += 1
            continue
        retained += 1
        condition_counts[item["condition"]] += 1
        model = item["models"]["arbert"]
        if model["mask_pos"] >= model["n_tokens"]:
            alignment_failures.append(item["id"])
    with (out / "arbert_coverage.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerow(["nominal_families", total])
        w.writerow(["retained_families", retained])
        w.writerow(["excluded_families", total - retained])
        w.writerow(["retained_alignment_failures", len(alignment_failures)])
        for cond in sorted(condition_counts):
            w.writerow([f"retained_{cond}", condition_counts[cond]])

    stats = read_csv(data / "stats_summary.csv")
    headline = [r for r in stats if r["note"] == ""]
    with (out / "headline_statistics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim", "model", "estimate", "ci_lo", "ci_hi", "p", "n"])
        for row in headline:
            w.writerow([row[k] for k in ["claim", "model", "estimate", "ci_lo", "ci_hi", "p", "n"]])

    natural = read_csv(data / "natural_behavioral.csv")
    natural_counts = Counter((r["model"], r["condition"]) for r in natural)
    with (out / "natural_counts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "rows"])
        for (model, condition), count in sorted(natural_counts.items()):
            w.writerow([model, condition, count])

    with (out / "release_audit.txt").open("w", encoding="utf-8") as f:
        f.write("IEEE Access release audit\n")
        f.write("=========================\n")
        f.write(f"ARBERT nominal families: {total}\n")
        f.write(f"ARBERT retained families: {retained}\n")
        f.write(f"ARBERT excluded families: {total - retained}\n")
        f.write(f"ARBERT retained alignment failures: {len(alignment_failures)}\n")
        f.write("All headline statistics copied from committed stats_summary.csv.\n")
        f.write("Natural counts copied from committed natural_behavioral.csv.\n")


if __name__ == "__main__":
    main()
