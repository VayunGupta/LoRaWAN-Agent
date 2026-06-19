# LoRaMAS: Multi-Agent Verification for LoRaWAN Security

Repository: https://github.com/Vayung2/LoRaWAN-Agent

This repository contains the implementation and paper source for LoRaMAS, a
server-side multi-agent verifier for LoRaWAN location-trust experiments. The
system evaluates real UVA LoRaWAN packet traces, injects controlled verification
attacks, and compares RF-only baselines against a supervisor that combines
gateway-local RF evidence, temporal packet-witness evidence, trilateration
residuals, and satellite/SAM-derived environmental context.

## Repository Layout

```text
dataset/                 UVA LoRaWAN metadata, packet traces, and weather files
models/                  Metadata and calibrated path-loss parameters
paper/                   LaTeX report, bibliography, and paper figures
scripts/                 Reproducibility scripts for results and figures
src/                     LoRaMAS implementation and baseline evaluators
src/react_agent/         Supervisor, specialist roles, tools, trust verifier
src/traditional/         Path-loss, trilateration, and calibration baselines
```

Generated outputs are written under `outputs/` and are intentionally ignored by
git. The SAM checkpoint is also ignored because it is a large downloaded model.

## Setup

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in repository already includes:

- `dataset/lorawan_metadata/*.parquet`
- `dataset/weather/deployment_weather.parquet`
- `models/metadata.json`
- `models/traditional_params.json`
- `paper/main.tex`
- `paper/refs.bib`

If metadata or calibration files need to be rebuilt:

```bash
python -m src.build_metadata
python -m src.traditional.calibrate \
  --data_dir dataset/lorawan_metadata \
  --meta models/metadata.json \
  --out models/traditional_params.json
```

## Quick Smoke Test

Run a clean LoRaMAS verification pass:

```bash
python -m src.run_react_agent \
  --attack-type none \
  --llm-mode off \
  --architecture loramas
```

Run one sensor-side attack:

```bash
python -m src.run_react_agent \
  --attack-type sensor_foil \
  --sensor sensor08 \
  --rssi-shift-db -12 \
  --llm-mode off \
  --architecture loramas \
  --json-out outputs/sensor08_sensor_foil_report.json
```

The command prints the supervisor decision, specialist-agent claims, cited
evidence keys, and the reasoning trace. The JSON file contains the same report
in machine-readable form.

## Main Experiment Commands

Run the paper-facing benchmark used for the MAS results table:

```bash
python -m src.evaluate_react_agent \
  --scenario-set benchmark \
  --benchmark-split all \
  --architectures localization_only centralized_trust loramas loramas_no_temporal loramas_no_physical \
  --role-reasoning rules \
  --modes off \
  --use-environment-context \
  --out-dir outputs/neurips_benchmark_rules
```

Important outputs:

- `outputs/neurips_benchmark_rules/react_eval_run_summary.csv`
- `outputs/neurips_benchmark_rules/react_eval_rows.csv`

Run representative single-case traces for manual inspection:

```bash
bash scripts/run_representative_results.sh
```

This writes human-readable traces to:

```text
outputs/logs/representative_results/
```

## Satellite/SAM Environment Context

The Environment Agent can use a cached satellite-context table. To build it,
download Meta's public SAM ViT-B checkpoint first:

```bash
mkdir -p models/sam
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o models/sam/sam_vit_b_01ec64.pth
```

Then run:

```bash
MPLCONFIGDIR=/private/tmp/mpl python scripts/build_satellite_context.py \
  --segmentation-mode sam \
  --sam-max-side 768
```

This writes:

- `outputs/satellite_context/uva_satellite.png`
- `outputs/satellite_context/satellite_context_by_pair.csv`
- `outputs/satellite_context/sam_building_mask.png`
- `outputs/satellite_context/sam_vegetation_mask.png`
- `outputs/satellite_context/satellite_los_overlay.png`
- `paper/satellite_los_overlay.png`

When `--use-environment-context` is enabled, the verifier reads
`outputs/satellite_context/satellite_context_by_pair.csv` if it exists. If that
file is missing, it falls back to the older OpenStreetMap-only context.

## One-Command Paper Result Bundle

To regenerate the localization summaries, localization-threshold detector,
satellite context, and MAS benchmark bundle used by `paper/main.tex`:

```bash
python scripts/run_main_tex_results.py --out-dir outputs/main_tex_results
```

For a fast sanity check:

```bash
python scripts/run_main_tex_results.py \
  --quick \
  --skip-satellite \
  --out-dir outputs/main_tex_results_quick
```

The full run writes a checklist here:

```text
outputs/main_tex_results/RESULTS_FOR_MAIN_TEX.md
```

## Compile the Paper

The report source is `paper/main.tex`.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled PDF is `paper/main.pdf`. Build artifacts such as `.aux`, `.bbl`,
`.blg`, `.log`, and `.pdf` are ignored by git; commit `main.tex`, `refs.bib`,
and the paper figures instead.

## LLM Modes

The default paper results use deterministic specialist-role rules. Optional LLM
support is available through a local Ollama endpoint:

- `--llm-mode off`: deterministic verifier only
- `--llm-mode explain`: deterministic label with optional LLM rationale
- `--llm-mode adjudicate`: bounded LLM override only on ambiguous cases
- `--role-reasoning llm`: rewrite specialist claims using the configured model

Example:

```bash
python -m src.run_react_agent \
  --attack-type random_noise \
  --sensor sensor05 \
  --noise-sigma-db 6 \
  --architecture loramas \
  --llm-mode adjudicate \
  --llm-model qwen2.5:7b
```

If Ollama is unavailable or returns malformed output, the implementation falls
back to the deterministic verifier.
 `requirements.txt`, `models/*.json`, dataset files,
  `paper/main.tex`, `paper/refs.bib`, and the paper figures used by LaTeX.
