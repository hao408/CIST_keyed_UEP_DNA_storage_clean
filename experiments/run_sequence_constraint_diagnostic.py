from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from common.image_io import load_rgb_image, prepare_images
from common.nature_style import apply_nature_style, clean_axis, save_figure
from common.security_features import (
    CONTAINER_BYTES,
    bytes_to_dna,
    keyed_bytes,
    macro_f1,
    max_homopolymer,
    payload_features,
    pseudo_parity_bytes,
    xor_bytes,
)
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


DESIGNS = [
    ("current_full_container_masking", 1, "Current full-mask"),
    ("resampling_8", 8, "Resampling 8"),
    ("resampling_16", 16, "Resampling 16"),
]
LEVELS = ["H", "M", "L"]
ATTACKERS = ["rule_based", "logistic_regression", "random_forest"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_violation(gc: float) -> float:
    if gc < 0.4:
        return 0.4 - gc
    if gc > 0.6:
        return gc - 0.6
    return 0.0


def penalty(gc: float, hp: int) -> float:
    return gc_violation(gc) + 0.1 * max(0, hp - 4)


def codeword_bytes(block, seed: int) -> bytes:
    return block.data + pseudo_parity_bytes(block.index, block.parity, seed)


def container_bytes_for_attempt(block, seed: int, attempt: int) -> tuple[bytes, bytes]:
    codeword = codeword_bytes(block, seed)
    if len(codeword) > CONTAINER_BYTES:
        raise ValueError("Effective codeword exceeds fixed container length")
    if attempt == 0:
        dummy_ns = "full-container-dummy"
        mask_ns = "full-container-mask"
    else:
        dummy_ns = f"constraint-dummy-attempt-{attempt}"
        mask_ns = f"constraint-mask-attempt-{attempt}"
    dummy = keyed_bytes(CONTAINER_BYTES - len(codeword), dummy_ns, block.index, seed)
    container = codeword + dummy
    mask = keyed_bytes(CONTAINER_BYTES, mask_ns, block.index, seed)
    return container, xor_bytes(container, mask)


def unmask_container(masked: bytes, block_index: int, seed: int, attempt: int) -> bytes:
    mask_ns = "full-container-mask" if attempt == 0 else f"constraint-mask-attempt-{attempt}"
    mask = keyed_bytes(CONTAINER_BYTES, mask_ns, block_index, seed)
    return xor_bytes(masked, mask)


def choose_payload(block, seed: int, max_attempts: int) -> dict:
    best = None
    for attempt in range(max_attempts):
        container, masked = container_bytes_for_attempt(block, seed, attempt)
        payload = bytes_to_dna(masked)
        gc = gc_content(payload)
        hp = max_homopolymer(payload)
        score = penalty(gc, hp)
        record = {
            "attempt": attempt,
            "container": container,
            "masked": masked,
            "payload": payload,
            "gc_content": gc,
            "max_homopolymer": hp,
            "penalty": score,
        }
        if best is None or score < best["penalty"]:
            best = record
        if 0.4 <= gc <= 0.6 and hp <= 4:
            best = record
            break
    assert best is not None
    unmasked = unmask_container(best["masked"], block.index, seed, best["attempt"])
    codeword = codeword_bytes(block, seed)
    roundtrip_ok = unmasked[: len(codeword)] == codeword
    best["roundtrip_ok"] = roundtrip_ok
    return best


def summarize_sequence(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["container_design"], row["max_attempts"])].append(row)
    out = []
    for (design, max_attempts), items in groups.items():
        vals_gc = [float(r["gc_content"]) for r in items]
        vals_hp = [int(r["max_homopolymer"]) for r in items]
        attempts = [int(r["selected_attempt"]) + 1 for r in items]
        out.append(
            {
                "container_design": design,
                "max_attempts": max_attempts,
                "num_payloads": len(items),
                "gc_mean": round(mean(vals_gc), 6),
                "gc_std": round(pstdev(vals_gc), 6),
                "max_homopolymer_mean": round(mean(vals_hp), 6),
                "max_homopolymer_std": round(pstdev(vals_hp), 6),
                "gc_pass_rate": round(mean(int(r["gc_pass"]) for r in items), 6),
                "hp4_pass_rate": round(mean(int(r["hp4_pass"]) for r in items), 6),
                "hp5_pass_rate": round(mean(int(r["hp5_pass"]) for r in items), 6),
                "combined_gc_hp4_pass_rate": round(mean(int(r["combined_gc_hp4_pass"]) for r in items), 6),
                "combined_gc_hp5_pass_rate": round(mean(int(r["combined_gc_hp5_pass"]) for r in items), 6),
                "mean_attempts": round(mean(attempts), 6),
                "max_attempt_failure_rate": round(mean(1 if int(r["selected_attempt"]) + 1 == int(max_attempts) and not int(r["combined_gc_hp4_pass"]) else 0 for r in items), 6),
                "roundtrip_mismatch_count": sum(1 for r in items if not int(r["roundtrip_ok"])),
                "stored_counter_bits_if_needed": 0 if int(max_attempts) <= 1 else math.ceil(math.log2(int(max_attempts))),
            }
        )
    return out


def table_sequence(summary_rows: list[dict], out_path: Path) -> None:
    labels = {
        "current_full_container_masking": "Current full-container masking",
        "resampling_8": "Constraint-aware resampling (8)",
        "resampling_16": "Constraint-aware resampling (16)",
    }
    lines = [
        "\\begin{center}",
        "\\captionof{table}{Payload-level sequence-constraint diagnostic for full-container masked fixed payloads. Constraints are computed on the 160-base payload only, excluding primers and index. Combined pass requires GC in [40\\%,60\\%] and maximum homopolymer length no greater than four. Resampling uses key-derived mask attempts and does not alter RS profiles, effective parity budget, payload length, or oligo length. This is a computational diagnostic, not biochemical optimization or wet-lab validation.}",
        "\\label{tab:sequence_constraint}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Container design & GC pass & HP$\\leq$4 pass & HP$\\leq$5 pass & Combined HP$\\leq$4 & Combined HP$\\leq$5 & Mean attempts & Failure rate & Round-trip mismatches \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{labels[row['container_design']]} & "
            f"{float(row['gc_pass_rate']):.4f} & {float(row['hp4_pass_rate']):.4f} & {float(row['hp5_pass_rate']):.4f} & "
            f"{float(row['combined_gc_hp4_pass_rate']):.4f} & {float(row['combined_gc_hp5_pass_rate']):.4f} & "
            f"{float(row['mean_attempts']):.3f} & {float(row['max_attempt_failure_rate']):.4f} & {int(row['roundtrip_mismatch_count'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{center}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_ml_dataset(rows: list[dict], design: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    feats = []
    labels = []
    groups = []
    payloads = []
    for row in rows:
        if row["container_design"] != design:
            continue
        payload = row["_payload"]
        feats.append(payload_features(payload))
        labels.append(row["level"])
        groups.append(f"{row['image']}|{row['seed']}")
        payloads.append(payload)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels), groups, payloads


def grouped_split(groups: list[str], split_seed: int, test_fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    unique = np.asarray(sorted(set(groups)))
    rng.shuffle(unique)
    n_test = max(1, int(round(test_fraction * len(unique))))
    test_groups = set(unique[:n_test])
    test = np.asarray([g in test_groups for g in groups], dtype=bool)
    train = ~test
    return train, test


def majority_predict(train_y: np.ndarray, test_n: int) -> np.ndarray:
    counts = Counter(train_y.tolist())
    majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return np.asarray([majority] * test_n)


def run_leakage(rows: list[dict], split_seeds: list[int]) -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = []
    for design, _, _ in DESIGNS:
        x, y, groups, _ = build_ml_dataset(rows, design)
        for split_seed in split_seeds:
            train, test = grouped_split(groups, split_seed)
            for attacker in ATTACKERS:
                start = time.perf_counter()
                if attacker == "rule_based":
                    pred = majority_predict(y[train], int(test.sum()))
                elif attacker == "logistic_regression":
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=500, class_weight="balanced", random_state=split_seed, n_jobs=1),
                    )
                    clf.fit(x[train], y[train])
                    pred = clf.predict(x[test])
                elif attacker == "random_forest":
                    clf = RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=split_seed,
                        n_jobs=-1,
                    )
                    clf.fit(x[train], y[train])
                    pred = clf.predict(x[test])
                else:
                    raise ValueError(attacker)
                truth = y[test]
                out.append(
                    {
                        "dataset": "custom_equal_budget",
                        "container_design": design,
                        "attacker_type": attacker,
                        "feature_set": "length_composition_tail_kmer_fixed_position_sequence_constraints",
                        "split_seed": split_seed,
                        "accuracy": round(float(accuracy_score(truth, pred)), 6),
                        "macro_f1": round(macro_f1(truth.tolist(), pred.tolist(), LEVELS), 6),
                        "num_train": int(train.sum()),
                        "num_test": int(test.sum()),
                        "runtime_seconds": round(time.perf_counter() - start, 6),
                    }
                )
    return out


def summarize_leakage(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["container_design"], row["attacker_type"], row["feature_set"])].append(row)
    out = []
    for (design, attacker, feature_set), items in groups.items():
        for metric in ["accuracy", "macro_f1"]:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "container_design": design,
                    "attacker_type": attacker,
                    "feature_set": feature_set,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_splits": len(items),
                    "num_runs": len(items),
                }
            )
    return out


def table_leakage(summary_rows: list[dict], out_path: Path) -> None:
    labels = {
        "current_full_container_masking": "Current full-container masking",
        "resampling_8": "Constraint-aware resampling (8)",
        "resampling_16": "Constraint-aware resampling (16)",
    }
    attackers = {"rule_based": "Rule-based", "logistic_regression": "Logistic regression", "random_forest": "Random forest"}
    lookup = defaultdict(dict)
    for row in summary_rows:
        lookup[(row["container_design"], row["attacker_type"])][row["metric"]] = (float(row["mean"]), float(row["std"]))
    lines = [
        "\\begin{center}",
        "\\captionof{table}{Protection-level leakage check after sequence-constraint-aware resampling. The task predicts the final H/M/L protection level from payload-level features. Values are mean $\\pm$ standard deviation across grouped split seeds 0--4. These results evaluate simulated observers and do not constitute a cryptographic proof.}",
        "\\label{tab:sequence_constraint_leakage}",
        "\\scriptsize",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Container design & Attacker & Accuracy & Macro-F1 \\\\",
        "\\midrule",
    ]
    for design, _, _ in DESIGNS:
        for attacker in ATTACKERS:
            acc = lookup[(design, attacker)]["accuracy"]
            f1 = lookup[(design, attacker)]["macro_f1"]
            lines.append(f"{labels[design]} & {attackers[attacker]} & {acc[0]:.4f} $\\pm$ {acc[1]:.4f} & {f1[0]:.4f} $\\pm$ {f1[1]:.4f} \\\\")
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_pass_rates(summary_rows: list[dict], out_stem: Path) -> None:
    apply_nature_style()
    labels = ["Current\nfull-mask", "Resampling\n8", "Resampling\n16"]
    metrics = [
        ("gc_pass_rate", "GC pass"),
        ("hp4_pass_rate", "HP<=4 pass"),
        ("combined_gc_hp4_pass_rate", "Combined pass"),
    ]
    x = np.arange(len(labels))
    width = 0.24
    colors = ["#6E7781", "#3F6FA8", "#2E8B73"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    for i, (metric, name) in enumerate(metrics):
        vals = [float(r[metric]) for r in summary_rows]
        ax.bar(x + (i - 1) * width, vals, width=width, label=name, color=colors[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass rate")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    clean_axis(ax)
    save_figure(fig, out_stem)
    plt.close(fig)


def plot_distributions(raw_rows: list[dict], out_stem: Path) -> None:
    apply_nature_style()
    labels = ["Current", "Resamp. 8", "Resamp. 16"]
    designs = [d[0] for d in DESIGNS]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), constrained_layout=True)
    gc_data = [[float(r["gc_content"]) for r in raw_rows if r["container_design"] == d] for d in designs]
    hp_data = [[int(r["max_homopolymer"]) for r in raw_rows if r["container_design"] == d] for d in designs]
    for ax, data, ylabel, panel in zip(axes, [gc_data, hp_data], ["GC content", "Max homopolymer"], ["a", "b"]):
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, widths=0.55)
        for patch, color in zip(bp["boxes"], ["#B8C0CC", "#9CB9D9", "#93C9B8"]):
            patch.set_facecolor(color)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.6)
        for item in bp["medians"]:
            item.set_color("#111111")
            item.set_linewidth(1.0)
        ax.set_ylabel(ylabel)
        ax.text(-0.18, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=8)
        clean_axis(ax)
    axes[0].axhspan(0.4, 0.6, color="#DDEFE6", alpha=0.5, zorder=-1)
    axes[1].axhline(4, color="#777777", linestyle="--", linewidth=0.8)
    save_figure(fig, out_stem)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-constraint-aware full-container masking diagnostic.")
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--resize", default="256,256")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    width, height = [int(x.strip()) for x in args.resize.split(",")]
    seeds = parse_seeds(args.seeds)
    split_seeds = parse_seeds(args.split_seeds)
    results_dir = args.paper_dir / "results" / "full_simulation"
    figures_dir = args.paper_dir / "figures" / "full_simulation"
    tables_dir = args.paper_dir / "tables" / "generated"
    rows: list[dict] = []
    start_all = time.perf_counter()
    for image_path in prepare_images(args.image_folder, args.paper_dir, False, (width, height)):
        img = load_rgb_image(image_path, (width, height))
        blocks, _, _ = decompose_bitplane_blocks(img, args.block_size)
        for seed in seeds:
            scheduled = apply_schedule(blocks, "keyed_uep", seed=seed)
            for block in scheduled:
                for design, max_attempts, _ in DESIGNS:
                    chosen = choose_payload(block, seed, max_attempts)
                    payload = chosen["payload"]
                    gc = float(chosen["gc_content"])
                    hp = int(chosen["max_homopolymer"])
                    row = {
                        "dataset": args.image_folder.name,
                        "image": image_path.name,
                        "seed": seed,
                        "block_index": block.index,
                        "channel": block.channel,
                        "bitplane": block.bitplane,
                        "base_level": block.base_level,
                        "level": block.level,
                        "parity": block.parity,
                        "container_design": design,
                        "max_attempts": max_attempts,
                        "selected_attempt": chosen["attempt"],
                        "payload_length_bases": 160,
                        "constraint_scope": "payload_only",
                        "gc_content": round(gc, 6),
                        "max_homopolymer": hp,
                        "gc_pass": int(0.4 <= gc <= 0.6),
                        "hp4_pass": int(hp <= 4),
                        "hp5_pass": int(hp <= 5),
                        "combined_gc_hp4_pass": int(0.4 <= gc <= 0.6 and hp <= 4),
                        "combined_gc_hp5_pass": int(0.4 <= gc <= 0.6 and hp <= 5),
                        "penalty": round(float(chosen["penalty"]), 6),
                        "roundtrip_ok": int(chosen["roundtrip_ok"]),
                        "_payload": payload,
                    }
                    rows.append(row)

    raw_out = [{k: v for k, v in row.items() if k != "_payload"} for row in rows]
    summary_rows = summarize_sequence(raw_out)
    write_csv(results_dir / "sequence_constraint_raw.csv", raw_out)
    write_csv(results_dir / "sequence_constraint_summary.csv", summary_rows)
    table_sequence(summary_rows, tables_dir / "table_sequence_constraint.tex")

    leakage_rows = run_leakage(rows, split_seeds)
    leakage_summary = summarize_leakage(leakage_rows)
    write_csv(results_dir / "sequence_constraint_leakage_raw.csv", leakage_rows)
    write_csv(results_dir / "sequence_constraint_leakage_summary.csv", leakage_summary)
    table_leakage(leakage_summary, tables_dir / "table_sequence_constraint_leakage.tex")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_pass_rates(summary_rows, figures_dir / "sequence_constraint_pass_rates")
    plot_distributions(raw_out, figures_dir / "sequence_constraint_gc_hp_distribution")

    print(f"Wrote {len(raw_out)} sequence rows")
    print(f"Wrote {len(leakage_rows)} leakage rows")
    print(f"Runtime seconds: {time.perf_counter() - start_all:.3f}")


if __name__ == "__main__":
    main()
