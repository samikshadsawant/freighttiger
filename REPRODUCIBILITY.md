# Reproducibility Test Report

- Result: **PASS**
- Test date (UTC): 2026-10-15T03:20:04.323820+00:00
- Pipeline command: `python -m src.run`
- Number of runs: 3
- Output: `outputs/final_output.csv`
- Candidate rows per run: 17 (expected)
- Route-week groups per run: 728 (expected)
- Comparison method: SHA-256 hashes of saved CSV files

## Run results

| Run | Saved output | SHA-256 |
|---|---|---|
| 1 | `outputs/reproducibility/run_1.csv` | `f57151a7b1ee16d91e4342e3ffd1080924536293458b0802ba2679db7b816413` |
| 2 | `outputs/reproducibility/run_2.csv` | `f57151a7b1ee16d91e4342e3ffd1080924536293458b0802ba2679db7b816413` |
| 3 | `outputs/reproducibility/run_3.csv` | `f57151a7b1ee16d91e4342e3ffd1080924536293458b0802ba2679db7b816413` |

## Conclusion

All three saved CSV files are byte-for-byte identical.

## LLM configuration

No generative LLM is used to generate verdicts. Note retrieval uses sentence embeddings, and verdicts are determined by explicit rules. LLM temperature is not applicable.
