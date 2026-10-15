# Reproducibility Test Report

- Result: **PASS**
- Test date (UTC): 2026-10-15T05:24:43.734053+00:00
- Pipeline command: `python -m src.run`
- Number of runs: 3
- Output: `outputs/final_output.csv`
- Candidate rows per run: 17 (expected)
- Route-week groups per run: 728 (expected)
- Comparison method: SHA-256 hashes of saved CSV files

## Run results

| Run | Saved output | SHA-256 |
|---|---|---|
| 1 | `outputs/reproducibility/run_1.csv` | `a2dfa88d3f3c20bb2f3dc961beae3042ff7171492af5e96a937fc3172a120277` |
| 2 | `outputs/reproducibility/run_2.csv` | `a2dfa88d3f3c20bb2f3dc961beae3042ff7171492af5e96a937fc3172a120277` |
| 3 | `outputs/reproducibility/run_3.csv` | `a2dfa88d3f3c20bb2f3dc961beae3042ff7171492af5e96a937fc3172a120277` |

## Conclusion

All three saved CSV files are byte-for-byte identical.

## LLM configuration

No generative LLM is used to generate verdicts. Note retrieval uses sentence embeddings, and verdicts are determined by explicit rules. LLM temperature is not applicable.
