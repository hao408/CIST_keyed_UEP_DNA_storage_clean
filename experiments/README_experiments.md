# Experiment Scripts

The scripts in this folder generate in-silico simulation outputs for the keyed
UEP image DNA-storage study. They should be run from the repository root.

This repository is simulation-only:

- no DNA synthesis;
- no sequencing;
- no wet-lab validation;
- no formal cryptographic proof.

For installation, dataset layout, and recommended manuscript-style commands,
see the top-level `README.md`.

## Main Experiments

```text
run_equal_budget_baselines.py
```
Equal-budget reliability comparison under simulated base substitution.

```text
run_bitplane_recovery.py
```
Bit-plane recovery analysis for bit-planes 7 through 0.

```text
run_schedule_key_dependence.py
```
Schedule-only diagnostic for same-key reproducibility, cross-key overlap, and
mean effective parity. This script does not run channel simulation.

```text
run_full_container_masking_pilot.py
```
Full-container leakage diagnostic. With `--result-scope full_simulation` and
`--prefix full_container_leakage`, it writes the final full-container leakage
tables used by the manuscript.

```text
run_full_container_reconstruction_and_rs8.py
```
Noiseless full-container round-trip checks, reconstruction consistency checks,
and Uniform RS-8 / all-high RS(40,32) full-parity upper-bound baseline.

```text
run_dna_channel_sweep.py
```
In-silico DNA-channel sweep across simplified substitution, indel, dropout,
coverage, and index-corruption models.

```text
run_larger_dataset_eval.py
```
Folder-based larger-dataset evaluation, e.g. Kodak 24 after the user supplies
benchmark images locally.

```text
run_sensitivity_analysis.py
```
One-factor-at-a-time sensitivity analysis for block size, parity profile, and
keyed perturbation ratio.

## Supplementary Diagnostics

```text
run_coverage_dropout_diagnostic.py
```
Oligo-level coverage/dropout robustness diagnostic.

```text
run_sequence_constraint_diagnostic.py
```
Payload-level sequence-constraint-aware full-container masking diagnostic.

```text
run_schedule_inference_attack.py
```
Schedule-inference diagnostic under simulated observer models.

```text
run_leakage_ml_attack.py
run_leakage_attack.py
```
Earlier protection-level leakage diagnostics retained for ablation and
reproducibility.

```text
run_dna_stress_channel.py
```
Sequence-property-dependent stress diagnostic. This is a boundary check, not a
wet-lab channel model.

```text
run_saliency_keyed_uep_pilot.py
run_saliency_alpha_scan.py
```
Optional pilot scripts for saliency-aware keyed scheduling. These are not the
default manuscript method.

## Common Modules

```text
common/
```
Reusable utilities for image loading, bit-plane decomposition, UEP scheduling,
RS-capacity simulation, simulated DNA channels, leakage features, metrics, and
Nature-style plotting.

## Outputs

By default, scripts write generated artifacts under:

```text
results/full_simulation/
figures/full_simulation/
tables/generated/
```

These generated files are ignored by Git in the clean code release.

