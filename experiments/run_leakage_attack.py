from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from random import Random
from statistics import mean, pstdev
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from common.dna_channel import bytes_to_dna
from common.experiment_core import add_common_args, figures_dir, parse_seeds, resize_tuple, results_dir
from common.image_io import load_rgb_image, prepare_images
from common.plotting import simple_bar_plot, write_csv
from common.uep_scheduler import apply_schedule, decompose_bitplane_blocks


DESIGNS = ["variable_length_payload", "fixed_deterministic_padding", "fixed_encrypted_dummy_padding"]
ATTACKERS = ["rule_based_attacker"]
LEVELS = ["H", "M", "L"]
CONTAINER_BASES = 160
DETERMINISTIC_PAD = "ACGT"


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def macro_f1(labels: list[str], preds: list[str]) -> float:
    scores = []
    for cls in LEVELS:
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def pseudo_parity_bytes(block_index: int, parity: int, seed: int) -> bytes:
    rng = Random(f"parity:{seed}:{block_index}:{parity}")
    return bytes(rng.randrange(256) for _ in range(parity))


def keyed_dummy_dna(length: int, block_index: int, seed: int) -> str:
    rng = Random(f"encrypted-dummy:{seed}:{block_index}:{length}")
    return "".join(rng.choice("ACGT") for _ in range(length))


def repeat_pad(length: int) -> str:
    return (DETERMINISTIC_PAD * ((length + len(DETERMINISTIC_PAD) - 1) // len(DETERMINISTIC_PAD)))[:length]


def make_payload(block, design: str, seed: int) -> str:
    codeword = block.data + pseudo_parity_bytes(block.index, block.parity, seed)
    effective_payload = bytes_to_dna(codeword)
    if design == "variable_length_payload":
        return effective_payload
    pad_len = max(0, CONTAINER_BASES - len(effective_payload))
    if design == "fixed_deterministic_padding":
        return effective_payload + repeat_pad(pad_len)
    if design == "fixed_encrypted_dummy_padding":
        return effective_payload + keyed_dummy_dna(pad_len, block.index, seed)
    raise ValueError(f"Unknown container design: {design}")


def trailing_deterministic_pad_len(payload: str) -> int:
    count = 0
    for i in range(len(payload) - 1, -1, -1):
        expected = DETERMINISTIC_PAD[i % len(DETERMINISTIC_PAD)]
        if payload[i] != expected:
            break
        count += 1
    return count


def rule_based_predict(payload: str, design: str, rng: Random) -> str:
    if design == "variable_length_payload":
        if len(payload) >= 160:
            return "H"
        if len(payload) >= 144:
            return "M"
        return "L"

    pad_tail = trailing_deterministic_pad_len(payload)
    if pad_tail >= 24:
        return "L"
    if pad_tail >= 16:
        return "M"
    if design == "fixed_deterministic_padding":
        return "H"

    # With a fixed-length container and no detectable deterministic padding,
    # this rule-based attacker has no level cue and falls back to random choice.
    return rng.choice(LEVELS)


def draw_confusion_matrix(path_png: Path, matrix: dict[tuple[str, str], int], title: str) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    cell = 90
    left = 120
    top = 90
    w = left + cell * 3 + 80
    h = top + cell * 3 + 80
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 25), title, fill="black")
    d.text((20, top + cell), "true", fill="black")
    max_count = max(matrix.values()) if matrix else 1
    for j, pred in enumerate(LEVELS):
        d.text((left + j * cell + 35, top - 30), pred, fill="black")
    for i, truth in enumerate(LEVELS):
        d.text((left - 40, top + i * cell + 35), truth, fill="black")
        for j, pred in enumerate(LEVELS):
            value = matrix.get((truth, pred), 0)
            shade = 255 - int(180 * value / max_count)
            x0 = left + j * cell
            y0 = top + i * cell
            d.rectangle([x0, y0, x0 + cell - 4, y0 + cell - 4], fill=(shade, shade, 255), outline="black")
            d.text((x0 + 28, y0 + 35), str(value), fill="black")
    img.save(path_png)
    pdf_path = path_png.with_suffix(".pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=(w, h))
    c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
    c.showPage()
    c.save()


def write_summary(rows: list[dict], out_path: Path) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["container_design"], row["attacker_type"])].append(row)
    out = []
    for (design, attacker), items in sorted(grouped.items()):
        for metric in ["accuracy", "macro_f1"]:
            vals = [float(r[metric]) for r in items]
            out.append(
                {
                    "container_design": design,
                    "attacker_type": attacker,
                    "metric": metric,
                    "mean": round(mean(vals), 6),
                    "std": round(pstdev(vals), 6) if len(vals) > 1 else 0.0,
                    "num_images": len(set(r["image"] for r in items)),
                    "num_seeds": len(set(r["seed"] for r in items)),
                    "num_runs": len(items),
                }
            )
    write_csv(out_path, out)
    return out


def write_latex_table(summary_rows: list[dict], out_path: Path) -> None:
    by_design = defaultdict(dict)
    for row in summary_rows:
        by_design[row["container_design"]][row["metric"]] = row
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Protection-level leakage attack under simulated payload/container observations. Results are generated by in-silico simulations and do not constitute a formal cryptographic security proof.}",
        r"\label{tab:protection_level_leakage}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Container design & Accuracy & Macro-F1 \\",
        r"\midrule",
    ]
    labels = {
        "variable_length_payload": "Variable length",
        "fixed_deterministic_padding": "Fixed deterministic padding",
        "fixed_encrypted_dummy_padding": "Fixed encrypted dummy padding",
    }
    for design in DESIGNS:
        acc = by_design[design]["accuracy"]
        f1 = by_design[design]["macro_f1"]
        lines.append(
            f"{labels[design]} & {float(acc['mean']):.4f} $\\pm$ {float(acc['std']):.4f} & "
            f"{float(f1['mean']):.4f} $\\pm$ {float(f1['std']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Protection-level leakage attack under simulated payload containers.")
    add_common_args(parser)
    parser.add_argument("--container-designs", default=",".join(DESIGNS))
    parser.add_argument("--methods", default="static_uep,keyed_uep")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = "1"
        resize = (64, 64)
    else:
        resize = resize_tuple(args.resize)

    rows = []
    cm_rows = []
    aggregate_cm = Counter()
    images = prepare_images(args.image_folder, args.output_dir, args.smoke, resize)
    designs = parse_list(args.container_designs)
    methods = parse_list(args.methods)
    for image_path in images:
        img = load_rgb_image(image_path, resize)
        blocks, _, _ = decompose_bitplane_blocks(img, args.block_size)
        for method in methods:
            for seed in parse_seeds(args.seeds):
                scheduled = apply_schedule(blocks, method, seed=seed)
                labels = [b.level for b in scheduled]
                for design in designs:
                    start = time.perf_counter()
                    rng = Random(f"attacker:{method}:{design}:{image_path.name}:{seed}")
                    payloads = [make_payload(b, design, seed) for b in scheduled]
                    preds = [rule_based_predict(payload, design, rng) for payload in payloads]
                    acc = sum(1 for y, p in zip(labels, preds) if y == p) / max(1, len(labels))
                    cm = Counter((y, p) for y, p in zip(labels, preds))
                    aggregate_cm.update({(design, y, p): c for (y, p), c in cm.items()})
                    cm_path = results_dir(args) / "protection_level_leakage_confusion_matrix.csv"
                    for truth in LEVELS:
                        for pred in LEVELS:
                            cm_rows.append(
                                {
                                    "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                                    "image": image_path.name,
                                    "method": method,
                                    "container_design": design,
                                    "attacker_type": "rule_based_attacker",
                                    "seed": seed,
                                    "true_level": truth,
                                    "predicted_level": pred,
                                    "count": cm.get((truth, pred), 0),
                                }
                            )
                    rows.append(
                        {
                            "dataset": "smoke" if args.smoke else (args.image_folder.name if args.image_folder else "custom"),
                            "image": image_path.name,
                            "method": method,
                            "container_design": design,
                            "attacker_type": "rule_based_attacker",
                            "seed": seed,
                            "accuracy": round(acc, 6),
                            "macro_f1": round(macro_f1(labels, preds), 6),
                            "confusion_matrix_path": str(cm_path),
                            "runtime_seconds": round(time.perf_counter() - start, 6),
                        }
                    )

    res = results_dir(args)
    figs = figures_dir(args)
    raw_path = res / "protection_level_leakage_raw.csv"
    summary_path = res / "protection_level_leakage_summary.csv"
    cm_path = res / "protection_level_leakage_confusion_matrix.csv"
    write_csv(raw_path, rows)
    write_csv(cm_path, cm_rows)
    summary = write_summary(rows, summary_path)

    acc_by_design = defaultdict(list)
    for row in rows:
        acc_by_design[row["container_design"]].append(float(row["accuracy"]))
    labels = DESIGNS
    values = [mean(acc_by_design[d]) for d in labels]
    simple_bar_plot(figs / "protection_level_leakage_accuracy.png", "protection_level_leakage_accuracy", labels, values, "attacker accuracy")

    fixed_matrix = {
        (truth, pred): aggregate_cm.get(("fixed_encrypted_dummy_padding", truth, pred), 0)
        for truth in LEVELS
        for pred in LEVELS
    }
    draw_confusion_matrix(figs / "protection_level_leakage_confusion_matrix.png", fixed_matrix, "fixed encrypted dummy padding")
    write_latex_table(summary, args.output_dir / "tables" / "generated" / "table_protection_level_leakage.tex")
    print(f"Wrote {len(rows)} rows to {raw_path}")


if __name__ == "__main__":
    main()
