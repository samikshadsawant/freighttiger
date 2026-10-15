# FreightTiger: Freight Cost Anomaly Detection

## 1. Overview

Requires **Python 3.10 or newer**; tested with Python 3.12.

FreightTiger identifies unusual freight costs by analyzing shipment records at the route-week level. It compares each route's cost per tonne-kilometre against its own historical performance and similar routes, then checks whether available context notes explain the increase.

The pipeline produces `outputs/final_output.csv`, containing candidate anomalies, their percentage increases, final flags, matched context-note IDs, and explanations.

## 2. How It Works

1. **Weekly aggregation:** Group shipments by route, route type, and Monday-based week.
2. **Cost calculation:** Calculate cost per tonne-kilometre as total freight cost divided by total tonne-kilometres.
3. **Baseline comparison:** Compare the current rate against the route's own history and other routes of the same route type in the same week.
4. **Candidate flagging:** Flag a route-week only when both percentage thresholds are exceeded and both baselines are available.
5. **Context retrieval:** Apply route and week-window eligibility first, encode note texts once, and rank eligible notes with `sentence-transformers/all-MiniLM-L6-v2`.
6. **Evidence-based verdict:** Assess every eligible note and prefer the highest-similarity one that passes the reviewed route, date, and evidence rules. Similarity alone never establishes justification.

## 3. Assumptions and Thresholds

### Own-history baseline

The baseline is the mean cost per tonne-kilometre of the previous **8 available weekly observations** for the same route. The current observation is excluded to prevent look-ahead bias. When fewer than eight prior observations exist, all available prior observations are used. The first observation has no historical baseline.

### Peer baseline

The peer baseline is the average cost per tonne-kilometre of other routes with the same route type during the same week. The current route is excluded. A peer baseline is unavailable when no other eligible route exists.

### Candidate thresholds

A route-week is selected when both conditions hold:

- More than **5% above its own-history baseline**.
- More than **15% above the peer baseline**.

The thresholds were selected to produce a practical review set while retaining the observed Mumbai-Pune drift. They are configurable heuristics, not statistically validated optimal cutoffs.

| Own-history threshold | Peer threshold | Candidate rows |
|---:|---:|---:|
| 5% | 15% | 17 |
| 10% | 15% | 7 |
| 5% | 10% | 23 |

Raising the own-history cutoff to 10% removes ten candidates, including early drift observations. Relaxing the peer cutoff to 10% adds six. The selected 5%/15% pair keeps the review set manageable while exposing the gradual relative rise.

## 4. Context Notes and Verdicts

The supplied dataset contains **10 context notes**. Notes are filtered by route applicability and date overlap before semantic ranking. The `can_justify_rise`, evidence category, reason summary, and applicability windows in `src/note_metadata.py` were reviewed and labeled by hand. A note can justify a rise only when its reviewed category is route-specific, its route and dates match, and it describes a cost-increasing cause. General cost pressure cannot explain one route's divergence from peers; notes describing normal conditions, unaffected costs, or unrelated routes are non-supporting. New notes remain unexplained until reviewed metadata is added.

The route/date gate matters more than semantic similarity because a highly related note can still describe the wrong corridor or a different week. Similarity measures topical relevance, not causal applicability; it ranks only evidence that has already passed the route and date gate.

- **Route-specific evidence:** N001 describes flooding on Chennai-Bangalore; N002 describes a festival surcharge on Ahmedabad-Mumbai. Both notes include reviewed applicability windows.
- **Nationwide evidence:** N003 describes a general diesel-price increase of roughly 5–7%. It does not, by itself, explain why one route is substantially more expensive than similar routes, so it never justifies one of these peer-screened candidates on its own.
- **Non-supporting evidence:** Notes that describe normal conditions, improved roads, unaffected costs, or unrelated routes must not justify an anomaly.

**N003 duration assumption:** Its metadata applicability window extends through 2025-12-31 because the note does not specify an end date. This is an implementation assumption, not a date explicitly stated in the source note.

**Weekly-window assumptions:** Notes with week-long events but no exact end date are assigned seven-day applicability windows by review, including non-supporting decoy notes such as N009. If no note is eligible, the explanation may mention a route-matching note within a seven-day near-miss gap; a recently ended note is preferred over a future decoy. A near-miss never justifies the rise.

If no eligible note adequately supports a candidate, the route-week remains flagged as unexplained.

## 5. Evaluation and Validation

The following checks have been run successfully:

- **18 automated tests passed**, covering weekly grouping, the cost formula and baselines, retrieval, unreviewed notes, end-to-end synthetic eligibility/ranking/support selection, output format, and the hand-labeled output set.
- **17/17 verdict test cases passed.** These manually defined cases cover valid evidence, Monday-based date boundaries, wrong routes, and non-supporting notes.
- **Retrieval evaluation:** 3/3 selected real cases retrieved the expected note; a synthetic example demonstrates a decoy ranking first while reviewed support is selected.
- **Output validation:** 17 rows and 8 columns passed the output-format checks.
- **Justification audit:** All 3 justified output rows passed the audit.

The verdict harness checks Monday-based date boundaries, including Jan 13 and Jan 27 around the N002 window. A separate hand-labeled CSV checks the verdict and matched-note ID for all 17 output rows. Unit tests cover formulas and baselines; the output numbers were independently recalculated during review. The justification audit rechecks all three justified rows. These checks do not establish 100% accuracy across all route-weeks, and the embedding evaluation remains small and deliberately selected.

The CSV is sorted by route alphabetically, then by week start chronologically. Of 17 candidates, three are justified and 14 unexplained. Eleven Mumbai-Pune observations between June and December 2025 show a gradual rise relative to peers, not 11 unrelated incidents; those flagged observations are not consecutive weeks. The rolling own-history baseline uses the prior eight available observations and can mask slow creep as its reference level drifts. Peer comparison helps surface that pattern but does not remove the limitation.

## 6. Reproducibility

The pipeline was run three times, producing 17 output rows from 728 route-week groups on each run.

All three saved CSV outputs had the same SHA-256 hash:

`a2dfa88d3f3c20bb2f3dc961beae3042ff7171492af5e96a937fc3172a120277`

The outputs were therefore byte-for-byte identical across those three runs.

Reproducibility was verified in the current environment. The embedding model must be available locally or downloadable for a fresh environment; Hugging Face network access can affect initial model setup.

## 7. Token and Cost Log

Usage statistics are recorded in `outputs/token_cost_log.json`.

| Metric | Recorded value |
|---|---:|
| Generative LLM calls | 0 |
| LLM input tokens | 0 |
| LLM output tokens | 0 |
| Estimated generative LLM cost | ₹0.00 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding encode calls | Recorded per run in `outputs/token_cost_log.json` |
| Tokens encoded | Counted with the model tokenizer per run |

The project uses embeddings for retrieval and explicit rules for verdicts rather than a generative LLM. This makes verdicts stable and repeatable, avoids generative API costs, and eliminates the risk of generated explanations inventing evidence. The zero cost applies only to generative LLM usage; embedding inference, compute, storage, and other infrastructure costs are not priced in the log.

## 8. Limitations

- Only 10 context notes are supplied, so relevant explanations may be missing.
- Hand-reviewed evidence categories and applicability windows can be incomplete or mistaken.
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
python -m scripts.eval_harness
```

Run retrieval evaluation and the semantic-decoy demonstration:

```powershell
python -m scripts.eval_retrieval
```

Validate the output and justification audit:

```powershell
python -m scripts.check_output
python -m scripts.audit_justified
python -m scripts.check_output_numbers
```

Inspect threshold sensitivity:

```powershell
python -m scripts.check_thresholds
```

Check reproducibility:

```powershell
python -m scripts.check_reproducibility
```

## 10. Outputs

- `outputs/weekly_metrics.csv` — aggregated route-week metrics and baselines.
- `outputs/final_output.csv` — candidate anomalies, verdicts, and explanations.
- `outputs/token_cost_log.json` — zero generative LLM usage plus measured embedding model, encode calls, and encoded tokens.
- `outputs/reproducibility/` — CSV outputs saved from the three reproducibility runs.
- `REPRODUCIBILITY.md` — reproducibility report.
