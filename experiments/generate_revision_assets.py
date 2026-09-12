from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common.experiment_core import resize_tuple
from common.image_io import load_rgb_image
from common.plotting import write_csv
from common.security_features import make_payload, max_homopolymer
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "CIST_keyed_perturbation_UEP_paper_draft"

METHOD_ORDER = [
    "no_rs",
    "uniform_rs4",
    "uniform_mixed_rs4_5",
    "static_uep",
    "random_budget_neutral_uep",
    "keyed_uep",
]

METHOD_LABELS = {
    "no_rs": "No RS",
    "uniform_rs4": "Uniform RS-4",
    "uniform_mixed_rs4_5": "Uniform RS-4/5",
    "static_uep": "Static UEP",
    "random_budget_neutral_uep": "Random UEP",
    "keyed_uep": "Keyed UEP",
}

CONTAINER_LABELS = {
    "variable_length_payload": "Variable length",
    "fixed_deterministic_padding": "Deterministic padding",
    "fixed_encrypted_dummy_padding": "Encrypted dummy",
}

ATTACKER_LABELS = {
    "rule_based_attacker": "Rule-based",
    "logistic_regression_attacker": "Logistic regression",
    "random_forest_attacker": "Random forest",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_pdf_png(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_pdf_png_many(fig: plt.Figure, bases: list[Path]) -> None:
    for base in bases:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc, ec, fontsize=7):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1F2933", linespacing=1.15)
    return patch


def arrow(ax, start, end, color="#4B5563"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.0, color=color, shrinkA=4, shrinkB=4),
    )


def draw_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 7.05)
    ax.axis("off")

    blue_fc, blue_ec = "#E7F0FA", "#2F5D9B"
    orange_fc, orange_ec = "#FFF0DD", "#C66A2B"
    green_fc, green_ec = "#EAF6EC", "#3F8F5A"
    gray_fc, gray_ec = "#F4F5F7", "#5B6470"
    callout_fc = "#FFF8ED"

    def lane(y: float, h: float, color: str, label: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (0.16, y),
                13.36,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                linewidth=0.6,
                facecolor=color,
                edgecolor="none",
                alpha=0.35,
                zorder=0,
            )
        )
        ax.text(0.36, y + h - 0.25, label, fontsize=8.4, weight="bold", color="#27364A", va="top")

    def flow(y: float, nodes: list[tuple[str, float, float, float, str, str]], color: str) -> dict[str, tuple[float, float]]:
        prev = None
        centers = {}
        for text, x, w, h, fc, ec in nodes:
            box(ax, (x, y), (w, h), text, fc, ec, fontsize=6.6)
            centers[text] = (x + w / 2, y + h / 2)
            if prev:
                arrow(ax, prev, (x, y + h / 2), color)
            prev = (x + w, y + h / 2)
        return centers

    lane(5.15, 1.72, "#DDEBFA", "Keyed stream generation")
    key_nodes = [
        ("Master key\n+ salt", 0.70, 1.32, 0.58, blue_fc, blue_ec),
        ("CIS-T stream\ngenerator", 2.45, 1.70, 0.58, blue_fc, blue_ec),
        ("$K^{\\mathrm{sch}}$: UEP\nscheduling", 4.65, 1.65, 0.58, blue_fc, blue_ec),
        ("$K^{\\mathrm{dif}}$: codeword\nmasking", 6.80, 1.85, 0.58, blue_fc, blue_ec),
    ]
    key_centers = flow(5.68, key_nodes, blue_ec)
    ax.text(
        9.15,
        5.96,
        "Coupled IS-skew-tent generator\n"
        "defined in this work",
        fontsize=6.2,
        color=blue_ec,
        va="center",
    )

    lane(2.80, 2.05, "#FFF1DF", "Importance-aware UEP coding")
    coding_nodes = [
        ("RGB\nimage", 0.70, 0.90, 0.60, gray_fc, gray_ec),
        ("Bit-plane\nblocks", 1.90, 1.15, 0.60, gray_fc, gray_ec),
        ("Importance\nlabels H/M/L", 3.35, 1.30, 0.60, orange_fc, orange_ec),
        ("Keyed UEP\nschedule", 4.98, 1.32, 0.60, orange_fc, orange_ec),
        ("Shortened RS profiles\nH: RS(40,32)\nM: RS(36,32)\nL: RS(34,32)", 6.65, 1.95, 0.86, orange_fc, orange_ec),
        ("Masked\ncodewords", 8.95, 1.18, 0.60, orange_fc, orange_ec),
    ]
    coding_centers = flow(3.55, coding_nodes, orange_ec)
    box(
        ax,
        (4.82, 2.93),
        (1.72, 0.42),
        "Budget-neutral\n10% promote / 20% demote",
        callout_fc,
        orange_ec,
        fontsize=5.8,
    )
    ax.text(5.68, 2.82, "average parity unchanged", fontsize=5.4, color=orange_ec, ha="center", va="top")
    arrow(ax, (5.48, 5.68), (5.64, 4.15), blue_ec)
    arrow(ax, (7.72, 5.68), (9.54, 4.15), blue_ec)

    lane(0.35, 1.92, "#E6F4EA", "Fixed-length DNA storage simulation")
    dna_nodes = [
        ("Fixed 160-base\ncontainer", 0.70, 1.45, 0.60, green_fc, green_ec),
        ("Encrypted\ndummy tail", 2.48, 1.25, 0.60, green_fc, green_ec),
        ("Index + primers", 4.05, 1.14, 0.60, green_fc, green_ec),
        ("208-nt\noligo", 5.50, 0.92, 0.60, green_fc, green_ec),
        ("Simulated DNA\nstorage channel", 6.72, 1.55, 0.60, gray_fc, gray_ec),
        ("Keyed decoding\n+ reconstruction", 8.62, 1.72, 0.60, gray_fc, gray_ec),
    ]
    flow(1.10, dna_nodes, green_ec)
    box(
        ax,
        (0.78, 0.49),
        (2.28, 0.38),
        "Raw H/M/L payloads\n160 / 144 / 136 bases",
        "#F5FBF6",
        green_ec,
        fontsize=5.6,
    )

    fig.tight_layout(pad=0.2)
    save_pdf_png_many(fig, [PAPER / "paper_assets" / "keyed_uep_framework", ROOT / "figures" / "full_simulation" / "keyed_uep_framework"])


def draw_equal_budget() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "equal_budget_baselines_raw.csv")
    rows = [r for r in rows if r["method"] in METHOD_ORDER]
    metrics = [
        ("psnr", "PSNR (dB)"),
        ("ssim", "SSIM"),
        ("important_block_recovery", "Important-block recovery"),
    ]
    rates = sorted({float(r["error_rate"]) for r in rows})
    colors = {
        "no_rs": "#9AA0A6",
        "uniform_rs4": "#6C8EBF",
        "uniform_mixed_rs4_5": "#9BB7D4",
        "static_uep": "#D9A441",
        "random_budget_neutral_uep": "#8FB996",
        "keyed_uep": "#4C78A8",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in METHOD_ORDER:
            values = []
            errors = []
            for rate in rates:
                vals = [float(r[metric]) for r in rows if r["method"] == method and float(r["error_rate"]) == rate]
                values.append(mean(vals))
                errors.append(pstdev(vals) if len(vals) > 1 else 0.0)
            ax.plot([r * 100 for r in rates], values, marker="o", markersize=3, linewidth=1.1, color=colors[method], label=METHOD_LABELS[method])
            ax.fill_between([r * 100 for r in rates], [v - e for v, e in zip(values, errors)], [v + e for v, e in zip(values, errors)], color=colors[method], alpha=0.10, linewidth=0)
        ax.set_xlabel("Base substitution rate (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([r * 100 for r in rates])
        ax.set_xticklabels(["0.5" if abs(r - 0.005) < 1e-12 else f"{r * 100:g}" for r in rates])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    axes[0].legend(loc="upper right", fontsize=6, ncol=1)
    fig.tight_layout(w_pad=1.0)
    save_pdf_png_many(
        fig,
        [
            PAPER / "paper_assets" / "equal_budget_baselines",
            ROOT / "figures" / "full_simulation" / "equal_budget_baselines",
        ],
    )


def draw_protection_leakage() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "protection_level_leakage_summary.csv")
    designs = ["variable_length_payload", "fixed_deterministic_padding", "fixed_encrypted_dummy_padding"]
    labels = ["Variable\nlength", "Fixed\ndeterministic", "Fixed encrypted\ndummy"]
    values = []
    errors = []
    for design in designs:
        matches = [r for r in rows if r["container_design"] == design and r["metric"] == "accuracy"]
        values.append(float(matches[0]["mean"]))
        errors.append(float(matches[0]["std"]))
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.bar(range(len(values)), values, yerr=errors, color=["#9AA0A6", "#D9A441", "#4C78A8"], edgecolor="white", linewidth=0.5, capsize=2)
    ax.axhline(1 / 3, color="#444444", linestyle="--", linewidth=0.8)
    ax.text(2.05, 1 / 3 + 0.03, "Random guess", fontsize=6, color="#444444")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rule-based attacker accuracy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Variable\nlength", "Deterministic\npadding", "Encrypted\ndummy"])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    save_pdf_png_many(fig, [PAPER / "paper_assets" / "protection_level_leakage_accuracy", ROOT / "figures" / "full_simulation" / "protection_level_leakage_accuracy"])


def draw_dna_channel_sweep() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "dna_channel_sweep_summary.csv")
    rows = [r for r in rows if r["method"] == "keyed_uep" and r["coverage"] == "30"]
    channel_order = [
        "base_substitution",
        "base_insertion",
        "base_deletion",
        "mixed_indel_substitution",
        "oligo_dropout",
        "coverage_consensus",
        "index_corruption",
    ]
    labels = {
        "base_substitution": "Base substitution",
        "base_insertion": "Base insertion",
        "base_deletion": "Base deletion",
        "mixed_indel_substitution": "Mixed indel + sub.",
        "oligo_dropout": "Oligo dropout",
        "coverage_consensus": "Coverage consensus",
        "index_corruption": "Index corruption",
    }
    colors = {
        "base_substitution": "#4C78A8",
        "base_insertion": "#9BB7D4",
        "base_deletion": "#D9A441",
        "mixed_indel_substitution": "#C96822",
        "oligo_dropout": "#8FB996",
        "coverage_consensus": "#5A9B69",
        "index_corruption": "#9AA0A6",
    }
    rates = sorted({float(r["error_rate"]) for r in rows})

    def plot_metric(metric: str, ylabel: str, filename: str, omit: set[str] | None = None) -> None:
        omit = omit or set()
        fig, ax = plt.subplots(figsize=(3.4, 2.35))
        for ch in channel_order:
            if ch in omit:
                continue
            values = []
            errors = []
            for rate in rates:
                matches = [r for r in rows if r["channel_type"] == ch and r["metric"] == metric and abs(float(r["error_rate"]) - rate) < 1e-12]
                if not matches:
                    continue
                values.append(float(matches[0]["mean"]))
                errors.append(float(matches[0]["std"]))
            if len(values) == len(rates):
                x = [r * 100 for r in rates]
                ax.plot(x, values, marker="o", markersize=2.8, linewidth=1.0, label=labels[ch], color=colors[ch])
                ax.fill_between(x, [v - e for v, e in zip(values, errors)], [v + e for v, e in zip(values, errors)], color=colors[ch], alpha=0.08, linewidth=0)
        ax.set_xlabel("Nominal error rate (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([r * 100 for r in rates])
        ax.set_xticklabels(["0.5" if abs(r - 0.005) < 1e-12 else f"{r * 100:g}" for r in rates])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        ax.legend(fontsize=5.2, loc="best")
        fig.tight_layout()
        save_pdf_png_many(fig, [PAPER / "paper_assets" / filename, ROOT / "figures" / "full_simulation" / filename])

    plot_metric("psnr", "PSNR (dB)", "dna_channel_sweep_psnr", omit={"coverage_consensus"})
    plot_metric("ssim", "SSIM", "dna_channel_sweep_ssim")
    plot_metric("important_block_recovery", "Important-block recovery", "dna_channel_sweep_recovery")


def draw_schedule_inference() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "schedule_inference_attack_summary.csv")
    designs = ["variable_length_payload", "fixed_deterministic_padding", "fixed_encrypted_dummy_padding"]
    attackers = ["rule_based_attacker", "logistic_regression_attacker", "random_forest_attacker"]
    colors = ["#9AA0A6", "#D9A441", "#4C78A8"]

    def val(task, design, attacker, metric):
        match = [
            r for r in rows
            if r["task"] == task and r["method"] == "keyed_uep" and r["container_design"] == design
            and r["attacker_type"] == attacker and r["metric"] == metric
        ][0]
        return float(match["mean"]), float(match["std"])

    for task, filename, title, random_line in [
        ("final_level", "schedule_inference_final_level", "Final protection-level inference", 1 / 3),
        ("medium_transition", "schedule_inference_medium_transition", "Medium-transition inference", None),
    ]:
        fig, ax = plt.subplots(figsize=(3.8, 2.35))
        x = list(range(len(designs)))
        width = 0.24
        for j, attacker in enumerate(attackers):
            vals, errs = zip(*[val(task, design, attacker, "accuracy") for design in designs])
            offset = (j - 1) * width
            ax.bar([i + offset for i in x], vals, width=width, yerr=errs, color=colors[j], label=ATTACKER_LABELS[attacker], edgecolor="white", linewidth=0.4, capsize=2)
        if random_line is not None:
            ax.axhline(random_line, color="#444444", linestyle="--", linewidth=0.8)
            ax.text(1.92, random_line + 0.035, "Random guess", fontsize=6, color="#444444")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(["Variable\nlength", "Deterministic\npadding", "Encrypted\ndummy"])
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        fig.tight_layout()
        save_pdf_png_many(fig, [PAPER / "paper_assets" / filename, ROOT / "figures" / "full_simulation" / filename])


def draw_dna_stress_channel() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "dna_stress_channel_summary.csv")
    rows = [r for r in rows if r["coverage"] == "30"]
    methods = ["uniform_rs4", "static_uep", "keyed_uep"]
    method_labels = {"uniform_rs4": "Uniform RS-4", "static_uep": "Static UEP", "keyed_uep": "Keyed UEP"}
    colors = {"uniform_rs4": "#9BB7D4", "static_uep": "#D9A441", "keyed_uep": "#4C78A8"}
    rates = sorted({float(r["error_rate"]) for r in rows})

    def metric_value(method, rate, metric):
        m = [r for r in rows if r["method"] == method and abs(float(r["error_rate"]) - rate) < 1e-12 and r["metric"] == metric][0]
        return float(m["mean"]), float(m["std"])

    for metric, ylabel, filename in [
        ("psnr", "PSNR (dB)", "dna_stress_channel_psnr"),
        ("important_block_recovery", "Important-block recovery", "dna_stress_channel_recovery"),
    ]:
        fig, ax = plt.subplots(figsize=(3.2, 2.25))
        for method in methods:
            vals, errs = zip(*[metric_value(method, rate, metric) for rate in rates])
            x = [r * 100 for r in rates]
            ax.plot(x, vals, marker="o", markersize=3, linewidth=1.1, color=colors[method], label=method_labels[method])
            ax.fill_between(x, [v - e for v, e in zip(vals, errs)], [v + e for v, e in zip(vals, errs)], color=colors[method], alpha=0.10, linewidth=0)
        ax.set_xlabel("Nominal error rate (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([r * 100 for r in rates])
        ax.set_xticklabels([f"{r * 100:g}" for r in rates])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        ax.legend(fontsize=6)
        fig.tight_layout()
        save_pdf_png_many(fig, [PAPER / "paper_assets" / filename, ROOT / "figures" / "full_simulation" / filename])


def dna_stats(seq: str) -> dict[str, float]:
    counts = Counter(seq)
    gc = (counts["G"] + counts["C"]) / max(1, len(seq))
    return {
        "length": len(seq),
        "gc_content": gc,
        "max_homopolymer": max_homopolymer(seq),
    }


def draw_hist(values: list[float], xlabel: str, base: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.hist(values, bins=30, color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of containers")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    save_pdf_png(fig, base)


def sequence_qc() -> None:
    image_dir = ROOT / "datasets" / "custom_equal_budget"
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    resize = (256, 256)
    per_seq = []
    for image_path in image_paths:
        image = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(image, 16)
        scheduled = apply_schedule(blocks, "keyed_uep", seed=0)
        for block in scheduled:
            payload = make_payload(block, "fixed_encrypted_dummy_padding", seed=0)
            stat = dna_stats(payload)
            per_seq.append(
                {
                    "dataset": "custom_equal_budget",
                    "image": image_path.name,
                    "method": "Keyed UEP",
                    "container_design": "fixed_encrypted_dummy_padding",
                    "sequence_type": "160-base payload container",
                    "sequence_length": stat["length"],
                    "gc_content": round(stat["gc_content"], 6),
                    "max_homopolymer": stat["max_homopolymer"],
                }
            )

    lengths = [int(r["sequence_length"]) for r in per_seq]
    gc_values = [float(r["gc_content"]) for r in per_seq]
    hp_values = [int(r["max_homopolymer"]) for r in per_seq]
    summary = [
        {
            "dataset": "custom_equal_budget",
            "num_images": len(image_paths),
            "num_sequences": len(per_seq),
            "sequence_type": "160-base payload container",
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": round(mean(lengths), 6),
            "mean_gc_content": round(mean(gc_values), 6),
            "std_gc_content": round(pstdev(gc_values), 6),
            "min_gc_content": round(min(gc_values), 6),
            "max_gc_content": round(max(gc_values), 6),
            "mean_max_homopolymer": round(mean(hp_values), 6),
            "max_homopolymer": max(hp_values),
            "fraction_outside_40_60_gc": round(sum(1 for x in gc_values if x < 0.40 or x > 0.60) / max(1, len(gc_values)), 6),
        }
    ]
    write_csv(ROOT / "results" / "full_simulation" / "sequence_qc_summary.csv", summary)
    draw_hist(gc_values, "GC content", ROOT / "figures" / "full_simulation" / "sequence_qc_gc_distribution", "#4C78A8")
    draw_hist(hp_values, "Maximum homopolymer length", ROOT / "figures" / "full_simulation" / "sequence_qc_homopolymer", "#D9A441")
    for suffix in [".pdf", ".png"]:
        for stem in ["sequence_qc_gc_distribution", "sequence_qc_homopolymer"]:
            src = ROOT / "figures" / "full_simulation" / f"{stem}{suffix}"
            dst = PAPER / "paper_assets" / f"{stem}{suffix}"
            dst.write_bytes(src.read_bytes())

    table_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Basic sequence-level diagnostics for fixed 160-base payload containers generated from the five representative RGB test images using Keyed UEP and encrypted dummy padding. These computational statistics do not constitute biochemical optimization, synthesis validation, or sequencing validation.}",
        r"\label{tab:sequence_qc}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Statistic & Value \\",
        r"\midrule",
    ]
    s = summary[0]
    entries = [
        ("Number of containers", f"{s['num_sequences']}"),
        ("Container length", f"{s['min_length']}--{s['max_length']} bases"),
        ("GC content", f"{s['mean_gc_content']:.4f} $\\pm$ {s['std_gc_content']:.4f}"),
        ("GC range", f"{s['min_gc_content']:.4f}--{s['max_gc_content']:.4f}"),
        ("Outside 40--60\\% GC", f"{s['fraction_outside_40_60_gc']:.4f}"),
        ("Mean max homopolymer", f"{s['mean_max_homopolymer']:.4f}"),
        ("Maximum homopolymer", f"{s['max_homopolymer']}"),
    ]
    for k, v in entries:
        table_lines.append(f"{k} & {v} \\\\")
    table_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    gen_table = ROOT / "tables" / "generated" / "table_sequence_qc.tex"
    gen_table.parent.mkdir(parents=True, exist_ok=True)
    gen_table.write_text("\n".join(table_lines), encoding="utf-8")
    (PAPER / "paper_tables" / "table_sequence_qc.tex").write_text("\n".join(table_lines), encoding="utf-8")


def split_schedule_tables() -> None:
    rows = read_csv(ROOT / "results" / "full_simulation" / "schedule_inference_attack_summary.csv")

    def metric_value(task: str, method: str, container: str, attacker: str, metric: str) -> str:
        vals = [
            r
            for r in rows
            if r["task"] == task
            and r["method"] == method
            and r["container_design"] == container
            and r["attacker_type"] == attacker
            and r["metric"] == metric
        ]
        if not vals:
            return "NA"
        r = vals[0]
        return f"{float(r['mean']):.4f} $\\pm$ {float(r['std']):.4f}"

    def write_table(task: str, label: str, caption: str, out_name: str) -> None:
        containers = ["variable_length_payload", "fixed_deterministic_padding", "fixed_encrypted_dummy_padding"]
        attackers = ["rule_based_attacker", "logistic_regression_attacker", "random_forest_attacker"]
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            caption,
            rf"\label{{{label}}}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Container & Attacker & Accuracy & Macro-F1 \\",
            r"\midrule",
        ]
        for container in containers:
            for attacker in attackers:
                acc = metric_value(task, "keyed_uep", container, attacker, "accuracy")
                f1 = metric_value(task, "keyed_uep", container, attacker, "macro_f1")
                lines.append(f"{CONTAINER_LABELS[container]} & {ATTACKER_LABELS[attacker]} & {acc} & {f1} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
        path = PAPER / "paper_tables" / out_name
        path.write_text("\n".join(lines), encoding="utf-8")

    write_table(
        "final_level",
        "tab:schedule_inference_final",
        r"\caption{Final protection-level inference for Keyed UEP under simulated container observations. Full per-method results are available in the generated CSV files.}",
        "table_schedule_inference_final.tex",
    )
    write_table(
        "medium_transition",
        "tab:schedule_inference_transition",
        r"\caption{Medium-transition inference for Keyed UEP under simulated container observations. Full per-method results are available in the generated CSV files.}",
        "table_schedule_inference_transition.tex",
    )


def main() -> None:
    setup_style()
    draw_framework()
    draw_equal_budget()
    draw_protection_leakage()
    draw_dna_channel_sweep()
    draw_schedule_inference()
    draw_dna_stress_channel()
    split_schedule_tables()


if __name__ == "__main__":
    main()
