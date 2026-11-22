"""Summarise paired simulated evacuation routes and observational warning overlays."""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np

METHODS = (
    ("nearest", "Nearest exit", "#526175"),
    ("forward", "Forward, immediate", "#70aeba"),
    ("forward_delay5", "Forward, 5 s delay", "#477d89"),
    ("fixed", "TabPFN, immediate", "#da8023"),
    ("fixed_delay5", "TabPFN, 5 s delay", "#b84942"),
)
DIFFUSIVITIES = (0.2, 0.4, 0.8, 1.6, 3.2)


def aggregate(rows, method):
    """Aggregate people within alarmed case variations; incomplete walks stay visible."""
    detected = [r for r in rows if r["alarm"] is not None]
    walks = [w for row in detected for w in row["routes"][method]]
    if not walks:
        raise ValueError("no alarmed route outcomes")
    completed = [w["walked_m"] for w in walks if w["completed"]]
    initial_and_walk = [
        (initial, walk)
        for row in detected
        for initial, walk in zip(row["initial_harm"], row["routes"][method], strict=True)
    ]
    return dict(
        cases=len(detected),
        people=len(walks),
        exposed=sum(w["exposed"] for w in walks),
        safe_at_alarm=sum(not initial for initial, _ in initial_and_walk),
        exposed_after_safe_start=sum(
            not initial and walk["exposed"] for initial, walk in initial_and_walk
        ),
        cases_any_exposed=sum(any(w["exposed"] for w in row["routes"][method]) for row in detected),
        mean_harm_s=float(np.mean([w["harmful_s"] for w in walks])),
        completed=len(completed),
        mean_completed_m=float(np.mean(completed)) if completed else None,
        beyond_horizon_s=sum(w["beyond_horizon_s"] for w in walks),
    )


def attach_initial_status(rows, settings):
    """Reconstruct each person's simulated harm at alarm without any model calls."""
    from tabpfn_spatial.hazard import hazard_from_fields
    from tabpfn_spatial.sim import Factory, simulate

    study = runpy.run_path(str(Path(__file__).with_name("route_study.py")))
    by_key = {(r["layout"], r["episode"], r["diffusion"]): r for r in rows}
    for name in settings["layouts"]:
        factory = (
            Factory.default()
            if name == "default"
            else Factory.random(np.random.default_rng(int(name.removeprefix("random"))))
        )
        for case in study["make_cases"](factory, settings["episodes"], settings["people"]):
            row = by_key[(name, case["index"], case["diffusion"])]
            if row["alarm"] != case["alarm"]:
                raise ValueError("reconstructed alarm does not match recorded case")
            if case["alarm"] is None:
                continue
            times = np.arange(0.0, case["alarm"] + 1.0)
            field = simulate(factory, [case["scenario"]], times, d=case["diffusion"])[0, -1]
            toxic, fire = hazard_from_fields(factory, field)
            row["initial_harm"] = [bool((toxic | fire)[factory.cell_of(p)]) for p in case["starts"]]


def overlay_counts(rows, method, threshold):
    """Pool advisory overlay flags along the routes actually followed."""
    walks = [w for row in rows if row["alarm"] is not None for w in row["routes"][method]]
    scores = [w["overlay"][threshold] for w in walks]
    keys = ("safe_s", "toxic_s", "true_positive_s", "false_positive_s", "false_negative_s")
    totals = {key: sum(s[key] for s in scores) for key in keys}
    totals["people_any_false_positive"] = sum(s["false_positive_s"] > 0 for s in scores)
    totals["people_any_false_negative"] = sum(s["false_negative_s"] > 0 for s in scores)
    return totals


def paired_interval(rows, method, reference):
    """Paired bootstrap of exposure-rate differences by base leak across layouts."""
    groups = []
    for episode in sorted({row["episode"] for row in rows}):
        group = [r for r in rows if r["episode"] == episode and r["alarm"] is not None]
        if not group:
            continue
        test = [w for r in group for w in r["routes"][method]]
        base = [w for r in group for w in r["routes"][reference]]
        if len(test) != len(base):
            raise ValueError("unpaired people")
        groups.append(
            (sum(w["exposed"] for w in test) - sum(w["exposed"] for w in base), len(test))
        )
    a = np.asarray(groups)
    rng = np.random.default_rng(8867)
    samples = a[rng.integers(len(a), size=(5000, len(a)))].sum(axis=1)
    return np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist()


def paired_counts(rows, method, reference, *, safe_start=False):
    """Count paired outcomes, optionally only for people initially in a safe cell."""
    counts = {"both": 0, "test_only": 0, "reference_only": 0, "neither": 0}
    for row in rows:
        if row["alarm"] is None:
            continue
        for initial, test, baseline in zip(
            row["initial_harm"], row["routes"][method], row["routes"][reference], strict=True
        ):
            if safe_start and initial:
                continue
            first, second = test["exposed"], baseline["exposed"]
            key = (
                "both"
                if first and second
                else "test_only"
                if first
                else "reference_only"
                if second
                else "neither"
            )
            counts[key] += 1
    return counts


def describe(rows):
    """Print route outcomes, paired uncertainty, overlay counts, and subgroups."""
    detected = [r for r in rows if r["alarm"] is not None]
    missing = [r for r in rows if r["alarm"] is None]
    print(
        "variants",
        len(rows),
        "alarmed",
        len(detected),
        "no alarm",
        len(missing),
        "no-alarm layout/leak pairs",
        len({(r["layout"], r["episode"]) for r in missing}),
    )
    metrics = {}
    for method, _, _ in METHODS:
        metrics[method] = aggregate(rows, method)
        print(
            method,
            json.dumps(metrics[method]),
            "paired exposed-rate delta vs nearest",
            paired_interval(rows, method, "nearest"),
        )
    for reference in ("nearest", "forward"):
        print(
            "fixed minus",
            reference,
            "paired exposed-rate interval",
            paired_interval(rows, "fixed", reference),
        )
        print(
            "fixed vs",
            reference,
            "paired people safe at alarm",
            json.dumps(paired_counts(rows, "fixed", reference, safe_start=True)),
        )
    print(
        "fixed delay5 minus immediate paired exposed-rate interval",
        paired_interval(rows, "fixed_delay5", "fixed"),
    )
    print(
        "fixed delay5 minus forward delay5 paired exposed-rate interval",
        paired_interval(rows, "fixed_delay5", "forward_delay5"),
    )
    print(
        "fixed delay5 vs forward delay5 paired people safe at alarm",
        json.dumps(paired_counts(rows, "fixed_delay5", "forward_delay5", safe_start=True)),
    )
    for threshold in ("0.19", "0.50"):
        print("fixed overlay", threshold, json.dumps(overlay_counts(rows, "fixed", threshold)))
    for key in ("layout", "diffusion"):
        for value in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == value]
            if any(r["alarm"] is not None for r in group):
                print(
                    key,
                    value,
                    "cases",
                    len(group),
                    json.dumps({m: aggregate(group, m) for m, _, _ in METHODS}),
                )
    for method in ("map_fixed", "map_forward"):
        values = [r[method] for r in detected]
        positive, negative = sum(v["positive"] for v in values), sum(v["negative"] for v in values)
        print(
            method,
            json.dumps(
                dict(
                    brier=float(np.mean([v["brier"] for v in values])),
                    recall=sum(v["tp"] for v in values) / positive if positive else None,
                    fpr=sum(v["fp"] for v in values) / negative if negative else None,
                )
            ),
        )
    return metrics


def draw(metrics, output):
    """Draw exposed-person proportion and completed-route distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(METHODS))
    labels = [label for _, label, _ in METHODS]
    colours = [colour for _, _, colour in METHODS]
    exposed = [100 * metrics[key]["exposed"] / metrics[key]["people"] for key, _, _ in METHODS]
    distances = [metrics[key]["mean_completed_m"] for key, _, _ in METHODS]
    bars_exposed = axes[0].bar(x, exposed, color=colours)
    bars_distance = axes[1].bar(x, distances, color=colours)
    axes[0].set_ylim(0, max(1.0, max(exposed) * 1.2))
    axes[0].bar_label(bars_exposed, labels=[f"{v:.1f}%" for v in exposed], padding=3, fontsize=8)
    axes[1].bar_label(bars_distance, labels=[f"{v:.1f}" for v in distances], padding=3, fontsize=8)
    axes[0].set_ylabel("People exposed after alarm (%)")
    axes[1].set_ylabel("Mean distance for completed walks (m)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired simulated evacuation routes", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "One alarm-time forecast · 5 s delay is assumed · no-alarm cases excluded\n"
        "Same simulator for contexts and truth · related people and leak variations",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    """Read complete results or draw the recorded summary without cloud calls."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, nargs="?")
    p.add_argument("--figure", type=Path)
    p.add_argument("--recorded", action="store_true")
    args = p.parse_args()
    if args.recorded:
        if args.input or not args.figure:
            p.error("--recorded requires a figure and no input")
        from route_recorded import METRICS

        draw(METRICS, args.figure)
        return
    if args.input is None:
        p.error("provide a case output file")
    metadata = json.loads(args.input.with_suffix(args.input.suffix + ".meta.json").read_text())
    settings = metadata["settings"]
    if metadata["seed"] != 20261203:
        p.error("unexpected scenario seed")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    expected = {
        (layout, i, d)
        for layout in settings["layouts"]
        for i in range(settings["episodes"])
        for d in DIFFUSIVITIES
    }
    actual = [(r["layout"], r["episode"], r["diffusion"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        p.error("missing or duplicated cases")
    attach_initial_status(rows, settings)
    metrics = describe(rows)
    if args.figure:
        draw(metrics, args.figure)


if __name__ == "__main__":
    main()
