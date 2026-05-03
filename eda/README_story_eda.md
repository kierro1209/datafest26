# Patient journey “predictability story” EDA

Scripts extend **[journey_eda.py](journey_eda.py)** and **[resource_pressure_eda.py](resource_pressure_eda.py)** on real encounter exports (`event_enriched` / `encounter_enriched`). They support slide narratives about **what / when / where** is predictable on longitudinal journeys—without claiming causal effects or true staffing capacity.

## Standard captions (denominator)

Figures that involve **next encounter** targets must state:

- **Linked pairs:**  
  `Denominator: linked encounters only (rows with an observed next encounter in this extract).`

- **Full cohort return-style plots:**  
  `Denominator: all encounters; terminal encounters (no observed next in extract) counted separately; note right-censoring near end of calendar coverage.`

Transition heatmaps add:

- **Empirical association only** — row-normalized conditional frequencies, not causal pathways.

## Leakage / censoring checklist (EDA + modeling)

- Use **current** encounter fields as inputs; **next\_*** columns are **labels only** for supervised targets.
- Do not label prediction plots as if linked-only slices were population-wide return rates.
- **Patient-level “latest” SDOH:** verify time scope before using as predictive inputs (per-encounter vs global latest).
- **Modeling:** split by patient; temporal validation on calendar when applicable.

## Commands

From the repo root (`datafest26/`):

```bash
# Journey plots including gap bins (token-aligned), specialty transitions, return refinements, pathway CSV
python eda/journey_eda.py \
  --input data/processed/event_enriched.csv.gz \
  --output-dir visuals/eda_story \
  --max-rows 500000

# Empirical top-1 / top-5 baselines (linked pairs)
python eda/next_event_baselines.py \
  --input data/processed/event_enriched.csv.gz \
  --output-dir visuals/eda_story

# Resource proxies + optional quadrant chart
python eda/resource_pressure_eda.py \
  --input data/processed/event_enriched.csv.gz \
  --output-dir visuals/eda_story/resource \
  --quadrant-chart
```

Or run **[run_story_eda.sh](run_story_eda.sh)** (adjust `--max-rows` / paths as needed).

## Outputs (typical)

| Artifact | Description |
|----------|-------------|
| `10_gap_bin_token_aligned_linked_pairs.png` | Gap bins match model `gap_ids` vocabulary |
| `11_specialty_next_specialty_transition_topK.png` | Specialty → next specialty (association) |
| `12_return_exclusive_bins_linked_pairs.png` | Mutually exclusive gap bins (linked) |
| `13_return_within_30d_simple_linked_pairs.png` | % next within 30d by group |
| `14_full_cohort_return_exclusive_bins.png` | Full cohort optional (terminal bucket) |
| `05b_*`, `11b_*` | If `--transition-max-gap-days D` set |
| `pathways_top_specialty_three_step.csv` | Top 3-step specialty chains |
| `baseline_topk_metrics.csv`, `baseline_topk_accuracy.png` | Empirical baselines |
| Resource dir | Monthly pressure CSVs + optional quadrant PNG |

Use **`--no-predictability-plots`** on `journey_eda.py` to skip figures 10–14 and pathway CSV for faster legacy-only runs.
