# FreightTiger: Freight Cost Anomaly Detection

## 1. Overview

FreightTiger identifies unusual freight costs by analyzing shipment records at the route-week level. It compares each route's cost per tonne-kilometre against its own historical performance and similar routes, then checks whether available context notes explain the increase.

The pipeline produces `outputs/final_output.csv`, containing candidate anomalies, their percentage increases, final flags, matched context-note IDs, and explanations.

## 2. How It Works

1. **Weekly aggregation:** Group shipments by route, route type, and Monday-based week.
2. **Cost calculation:** Calculate cost per tonne-kilometre as total freight cost divided by total tonne-kilometres.
3. **Baseline comparison:** Compare the current rate against the route's own history and other routes of the same route type in the same week.
4. **Candidate flagging:** Flag a route-week only when both percentage thresholds are exceeded and both baselines are available.
5. **Context retrieval:** Use `sentence-transformers/all-MiniLM-L6-v2` to rank eligible context notes by semantic similarity.
6. **Evidence-based verdict:** Apply explicit rules to check route applicability, date overlap, and whether a note can justify a cost increase. Semantic similarity alone never establishes justification.

## 3. Assumptions and Thresholds

### Own-history baseline

The baseline is the mean cost per tonne-kilometre of the previous **8 available weekly observations** for the same route. The current observation is excluded to prevent look-ahead bias. When fewer than eight prior observations exist, all available prior observations are used. The first observation has no historical baseline.

### Peer baseline

The peer baseline is the average cost per tonne-kilometre of other routes with the same route type during the same week. The current route is excluded. A peer baseline is unavailable when no other eligible route exists.

### Candidate thresholds

A route-week is selected when both conditions hold:

- More than **5% above its own-history baseline**.
- More than **15% above the peer baseline**.

The thresholds were selected after examining the distributions of the percentage comparisons and reviewing the resulting candidate count. They are configurable heuristic thresholds, not statistically validated optimal cutoffs.

## 4. Context Notes and Verdicts

The supplied dataset contains **10 context notes**. Notes are filtered by route applicability and date overlap before semantic ranking.

- **Route-specific evidence:** N001 describes flooding on Chennai-Bangalore; N002 describes a festival surcharge on Ahmedabad-Mumbai.
- **Nationwide evidence:** N003 describes a general diesel-price increase of roughly 5–7%. It does not, by itself, explain why one route is substantially more expensive than similar routes.
- **Non-supporting evidence:** Notes that describe normal conditions, improved roads, unaffected costs, or unrelated routes must not justify an anomaly.

**N003 duration assumption:** Its metadata applicability window extends through 2025-12-31 because the note does not specify an end date. This is an implementation assumption, not a date explicitly stated in the source note.

If no eligible note adequately supports a candidate, the route-week remains flagged as unexplained.

## 5. Evaluation and Validation

The following checks have been run successfully:

- **13 automated unit and retrieval tests passed**, covering weekly grouping, the cost formula, historical look-ahead prevention, peer exclusion, note eligibility, and selected semantic-ranking cases.
- **16/16 verdict test cases passed.** These manually defined cases cover valid evidence, wrong routes, invalid dates, and non-supporting notes.
- **Initial retrieval evaluation:** 3/3 selected cases retrieved the expected top-ranked note and produced the expected verdict.
- **Output validation:** 17 rows and 8 columns passed the output-format checks.
- **Justification audit:** All 3 justified output rows passed the audit.

These results demonstrate correctness on the tested cases; they do not establish 100% accuracy across all route-weeks. The retrieval evaluation uses a small, deliberately selected test set.

## 6. Reproducibility

The pipeline was run three times, producing 17 output rows from 728 route-week groups on each run.

All three saved CSV outputs had the same SHA-256 hash:

`f57151a7b1ee16d91e4342e3ffd1080924536293458b0802ba2679db7b816413`

The outputs were therefore byte-for-byte identical across those three runs.

Reproducibility was verified in the current environment. The embedding model must be available locally or downloadable for a fresh environment; Hugging Face network access can affect initial model setup.

## 7. Token and Cost Log

Usage statistics are recorded in `outputs/token_cost_log.json`.

| Metric | Recorded value |
|---|---:|
| Generative LLM calls | 0 |
| LLM input tokens | 0 |
| LLM output tokens | 0 |
| Estimated LLM cost | ₹0.00 |

The project uses an embedding model for semantic retrieval and rule-based verdicts rather than a generative LLM. The recorded zero cost applies to generative LLM usage only; it does not include embedding inference, compute, storage, or other infrastructure costs.

## 8. Limitations

- Only 10 context notes are supplied, so relevant explanations may be missing.
- Retrieval and verdict accuracy have not been measured on a large, independently labelled evaluation dataset.
- The 5% and 15% thresholds are heuristics and may require recalibration on new data.
- N003's end date is assumed rather than explicitly provided.
- An unexplained increase is not proof of fraud or an incorrect charge; it indicates that the available evidence does not adequately explain the increase.
- The pipeline identifies candidates only when both baselines are available and both thresholds are exceeded.

## 9. Running the Project

Run commands from the project root with the virtual environment activated.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run anomaly detection:

```powershell
python -m src.run
```

Run automated tests:

```powershell
python -m pytest tests/ -v
```

Run the verdict evaluation:

```powershell
python eval_harness.py
```

Validate the output and justification audit:

```powershell
python check_output.py
python audit_justified.py
```

Check reproducibility:

```powershell
python check_reproducibility.py
```

## 10. Outputs

- `outputs/weekly_metrics.csv` — aggregated route-week metrics and baselines.
- `outputs/final_output.csv` — candidate anomalies, verdicts, and explanations.
- `outputs/token_cost_log.json` — generative LLM usage and estimated cost.
- `outputs/reproducibility/` — CSV outputs saved from the three reproducibility runs.
- `REPRODUCIBILITY.md` — reproducibility report.
