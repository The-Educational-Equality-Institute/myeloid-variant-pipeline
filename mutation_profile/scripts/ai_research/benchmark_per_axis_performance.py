"""Per-axis sensitivity/specificity for benchmark pathogenicity scoring tools.

Uses ClinVar as ground truth across 68 variants from benchmark_results.json.
Computes sensitivity, specificity, PPV, NPV, and coverage for each tool.

Ground truth:
  - Positive (pathogenic): ClinVar Pathogenic or Likely_Pathogenic
  - Negative (benign/uncertain): ClinVar VUS (no B/LB in this dataset)

Thresholds (from benchmark pipeline):
  - AlphaMissense >= 0.564 -> pathogenic
  - EVE >= 0.5 -> pathogenic
  - CADD >= 17.3 -> pathogenic
  - REVEL >= 0.5 -> pathogenic
  - SIFT <= 0.05 -> pathogenic (inverted scale)
  - PolyPhen-2 >= 0.453 -> pathogenic
"""

import json
from pathlib import Path
from datetime import datetime, timezone

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ai_research" / "benchmark"
INPUT_FILE = BENCHMARK_DIR / "benchmark_results.json"
OUTPUT_JSON = BENCHMARK_DIR / "per_axis_performance.json"
OUTPUT_MD = BENCHMARK_DIR / "per_axis_performance.md"

THRESHOLDS = {
    "alphamissense": {"threshold": 0.564, "direction": "gte", "label": "AlphaMissense"},
    "eve_score": {"threshold": 0.5, "direction": "gte", "label": "EVE"},
    "cadd_phred": {"threshold": 17.3, "direction": "gte", "label": "CADD"},
    "revel": {"threshold": 0.5, "direction": "gte", "label": "REVEL"},
    "sift_score": {"threshold": 0.05, "direction": "lte", "label": "SIFT"},
    "polyphen2_score": {"threshold": 0.453, "direction": "gte", "label": "PolyPhen-2"},
}

POSITIVE_CLASSES = {"Pathogenic", "Likely_Pathogenic"}
NEGATIVE_CLASSES = {"VUS"}


def classify_by_tool(score, threshold, direction):
    """Return True if tool predicts pathogenic."""
    if direction == "gte":
        return score >= threshold
    return score <= threshold


def compute_metrics(tp, fp, tn, fn):
    """Compute sensitivity, specificity, PPV, NPV from confusion matrix."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else None
    return {
        "sensitivity": round(sensitivity, 4) if sensitivity is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "ppv": round(ppv, 4) if ppv is not None else None,
        "npv": round(npv, 4) if npv is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    clinvar_variants = [
        v for v in data["variants"]
        if v["clinvar"]["normalized"] in POSITIVE_CLASSES | NEGATIVE_CLASSES
    ]

    n_total_clinvar = len(clinvar_variants)
    n_positive = sum(1 for v in clinvar_variants if v["clinvar"]["normalized"] in POSITIVE_CLASSES)
    n_negative = sum(1 for v in clinvar_variants if v["clinvar"]["normalized"] in NEGATIVE_CLASSES)

    print(f"ClinVar variants: {n_total_clinvar} ({n_positive} P/LP, {n_negative} VUS)")

    results = {}

    for tool_key, tool_cfg in THRESHOLDS.items():
        label = tool_cfg["label"]
        threshold = tool_cfg["threshold"]
        direction = tool_cfg["direction"]

        scored_variants = [v for v in clinvar_variants if v["scores"][tool_key] is not None]
        coverage = len(scored_variants)

        tp = fp = tn = fn = 0
        misclassified = []

        for v in scored_variants:
            score = v["scores"][tool_key]
            is_clinvar_positive = v["clinvar"]["normalized"] in POSITIVE_CLASSES
            tool_predicts_pathogenic = classify_by_tool(score, threshold, direction)

            if is_clinvar_positive and tool_predicts_pathogenic:
                tp += 1
            elif is_clinvar_positive and not tool_predicts_pathogenic:
                fn += 1
                misclassified.append({
                    "gene": v["gene"],
                    "variant": v["hgvsp"],
                    "score": score,
                    "clinvar": v["clinvar"]["normalized"],
                    "error": "false_negative",
                })
            elif not is_clinvar_positive and tool_predicts_pathogenic:
                fp += 1
                misclassified.append({
                    "gene": v["gene"],
                    "variant": v["hgvsp"],
                    "score": score,
                    "clinvar": v["clinvar"]["normalized"],
                    "error": "false_positive",
                })
            else:
                tn += 1

        metrics = compute_metrics(tp, fp, tn, fn)

        scored_positive = sum(1 for v in scored_variants if v["clinvar"]["normalized"] in POSITIVE_CLASSES)
        scored_negative = sum(1 for v in scored_variants if v["clinvar"]["normalized"] in NEGATIVE_CLASSES)

        results[tool_key] = {
            "label": label,
            "threshold": threshold,
            "direction": "greater_or_equal" if direction == "gte" else "less_or_equal",
            "coverage": {
                "scored": coverage,
                "total_clinvar": n_total_clinvar,
                "fraction": round(coverage / n_total_clinvar, 4) if n_total_clinvar > 0 else 0,
                "scored_positive": scored_positive,
                "scored_negative": scored_negative,
            },
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "metrics": metrics,
            "misclassified": misclassified,
        }

        print(f"\n{label} (threshold={threshold}, dir={direction}):")
        print(f"  Coverage: {coverage}/{n_total_clinvar} ({scored_positive} P/LP, {scored_negative} VUS)")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Sensitivity={metrics['sensitivity']}  Specificity={metrics['specificity']}")
        print(f"  PPV={metrics['ppv']}  NPV={metrics['npv']}  Accuracy={metrics['accuracy']}")
        if misclassified:
            print(f"  Misclassified ({len(misclassified)}):")
            for m in misclassified:
                print(f"    {m['gene']} {m['variant']}: {m['error']} (score={m['score']}, ClinVar={m['clinvar']})")

    output = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source": "benchmark_results.json",
            "method": "Per-axis sensitivity/specificity using ClinVar as ground truth",
            "ground_truth": {
                "positive": list(POSITIVE_CLASSES),
                "negative": list(NEGATIVE_CLASSES),
            },
            "total_variants": len(data["variants"]),
            "clinvar_variants": n_total_clinvar,
            "clinvar_positive": n_positive,
            "clinvar_negative": n_negative,
        },
        "tools": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {OUTPUT_JSON}")

    generate_markdown(output)
    print(f"Markdown saved: {OUTPUT_MD}")


def fmt(val, pct=False):
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if pct:
        return f"{val * 100:.1f}%"
    return f"{val:.4f}"


def generate_markdown(output):
    """Generate markdown report from results."""
    meta = output["metadata"]
    tools = output["tools"]

    lines = [
        "# Per-Axis Pathogenicity Tool Performance",
        "",
        f"Generated: {meta['generated'][:10]}",
        "",
        "## Method",
        "",
        f"Ground truth: ClinVar classifications from {meta['clinvar_variants']} variants "
        f"({meta['clinvar_positive']} Pathogenic/Likely Pathogenic, {meta['clinvar_negative']} VUS).",
        "",
        "Each tool is evaluated independently using its standard pathogenicity threshold. "
        "A tool call is a true positive if it predicts pathogenic for a ClinVar P/LP variant, "
        "and a true negative if it predicts benign/uncertain for a ClinVar VUS variant.",
        "",
        "## Summary Table",
        "",
        "| Tool | Threshold | Coverage | Sensitivity | Specificity | PPV | NPV | Accuracy |",
        "|------|-----------|----------|-------------|-------------|-----|-----|----------|",
    ]

    for tool_key in THRESHOLDS:
        t = tools[tool_key]
        m = t["metrics"]
        cov = t["coverage"]
        direction_symbol = ">=" if t["direction"] == "greater_or_equal" else "<="
        threshold_str = f"{direction_symbol} {t['threshold']}"
        coverage_str = f"{cov['scored']}/{cov['total_clinvar']} ({fmt(cov['fraction'], pct=True)})"

        lines.append(
            f"| {t['label']} | {threshold_str} | {coverage_str} "
            f"| {fmt(m['sensitivity'], pct=True)} | {fmt(m['specificity'], pct=True)} "
            f"| {fmt(m['ppv'], pct=True)} | {fmt(m['npv'], pct=True)} "
            f"| {fmt(m['accuracy'], pct=True)} |"
        )

    lines.extend([
        "",
        "## Confusion Matrices",
        "",
    ])

    for tool_key in THRESHOLDS:
        t = tools[tool_key]
        cm = t["confusion_matrix"]
        cov = t["coverage"]

        if cov["scored"] == 0:
            lines.extend([
                f"### {t['label']}",
                "",
                "No scored variants available (0% coverage).",
                "",
            ])
            continue

        lines.extend([
            f"### {t['label']}",
            "",
            f"Coverage: {cov['scored']}/{cov['total_clinvar']} variants "
            f"({cov['scored_positive']} P/LP, {cov['scored_negative']} VUS)",
            "",
            "| | Predicted Pathogenic | Predicted Benign/Uncertain |",
            "|---|---|---|",
            f"| ClinVar P/LP | TP = {cm['tp']} | FN = {cm['fn']} |",
            f"| ClinVar VUS | FP = {cm['fp']} | TN = {cm['tn']} |",
            "",
        ])

        if t["misclassified"]:
            fn_list = [m for m in t["misclassified"] if m["error"] == "false_negative"]
            fp_list = [m for m in t["misclassified"] if m["error"] == "false_positive"]

            if fn_list:
                lines.append(f"**False negatives** ({len(fn_list)} P/LP variants missed):")
                lines.append("")
                for m in fn_list:
                    lines.append(f"- {m['gene']} {m['variant']}: score = {m['score']} ({m['clinvar']})")
                lines.append("")

            if fp_list:
                lines.append(f"**False positives** ({len(fp_list)} VUS variants called pathogenic):")
                lines.append("")
                for m in fp_list:
                    lines.append(f"- {m['gene']} {m['variant']}: score = {m['score']} ({m['clinvar']})")
                lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- **Sensitivity** (recall): proportion of true P/LP variants the tool correctly flags as pathogenic. "
        "Higher = fewer missed pathogenic variants.",
        "- **Specificity**: proportion of true VUS/B/LB variants the tool correctly identifies as non-pathogenic. "
        "Higher = fewer false alarms.",
        "- **PPV** (precision): when the tool says pathogenic, how often is it correct?",
        "- **NPV**: when the tool says non-pathogenic, how often is it correct?",
        "- **Coverage**: fraction of ClinVar variants that have a score from this tool. "
        "Tools with 0% coverage (CADD, PolyPhen-2) could not be evaluated.",
        "",
        "The multi-axis Bayesian framework (Tavtigian 2018) combines these tools to reduce "
        "individual tool weaknesses. Tools with high sensitivity but lower specificity (e.g., SIFT, REVEL) "
        "complement tools with higher specificity but lower coverage (e.g., EVE).",
    ])

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
