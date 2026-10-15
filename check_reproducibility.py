
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_PATH = Path("outputs/final_output.csv")
EVIDENCE_DIR = Path("outputs/reproducibility")
REPORT_PATH = Path("REPRODUCIBILITY.md")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    hashes = []
    saved_outputs = []

    for run_number in range(1, 4):
        print(f"\n{'=' * 12} RUN {run_number}/3 {'=' * 12}")

        subprocess.run(
            [sys.executable, "-m", "src.run"],
            check=True,
        )

        if not OUTPUT_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline did not create {OUTPUT_PATH}"
            )

        saved_path = EVIDENCE_DIR / f"run_{run_number}.csv"
        shutil.copy2(OUTPUT_PATH, saved_path)

        digest = file_hash(saved_path)
        hashes.append(digest)
        saved_outputs.append(saved_path)

        print(f"Saved: {saved_path}")
        print(f"SHA-256: {digest}")

    identical = len(set(hashes)) == 1
    status = "PASS" if identical else "FAIL"

    report_lines = [
        "# Reproducibility Test Report",
        "",
        f"- Result: **{status}**",
        f"- Test date (UTC): {datetime.now(timezone.utc).isoformat()}",
        "- Pipeline command: `python -m src.run`",
        "- Number of runs: 3",
        "- Output: `outputs/final_output.csv`",
        "- Candidate rows per run: 17 (expected)",
        "- Route-week groups per run: 728 (expected)",
        "- Comparison method: SHA-256 hashes of saved CSV files",
        "",
        "## Run results",
        "",
        "| Run | Saved output | SHA-256 |",
        "|---|---|---|",
    ]

    for index, (path, digest) in enumerate(
        zip(saved_outputs, hashes), start=1
    ):
        report_lines.append(f"| {index} | `{path.as_posix()}` | `{digest}` |")

    report_lines.extend([
        "",
        "## Conclusion",
        "",
        (
            "All three saved CSV files are byte-for-byte identical."
            if identical
            else "The output files differ. Investigate the differences."
        ),
        "",
        "## LLM configuration",
        "",
        (
            "No generative LLM is used to generate verdicts. "
            "Note retrieval uses sentence embeddings, and verdicts "
            "are determined by explicit rules. LLM temperature is "
            "not applicable."
        ),
        "",
    ])

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\nReport saved: {REPORT_PATH}")

    if not identical:
        raise SystemExit(
            "FAIL: Outputs differ. See the saved CSV files and report."
        )

    print("PASS: All three saved outputs are byte-for-byte identical.")


if __name__ == "__main__":
    main()
