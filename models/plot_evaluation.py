from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "models" / "transformer_evaluation.csv"
OUTPUT_DIR = ROOT / "models"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {CSV_FILE}\n"
        "Run evaluate_transformer.py first."
    )

df = pd.read_csv(CSV_FILE)

required = {"reference", "prediction", "exact_match", "cer", "wer"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

# 1. Exact-match performance
accuracy = df["exact_match"].astype(float).mean() * 100

plt.figure(figsize=(6, 4))
plt.bar(["Case-insensitive exact match", "Not exact match"],
        [accuracy, 100 - accuracy])
plt.ylabel("Percentage (%)")
plt.title("TrOCR Test-Set Exact-Match Performance")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "exact_match_chart.png", dpi=200)
plt.close()

# 2. Character and word error rates
plt.figure(figsize=(6, 4))
plt.bar(["Mean CER", "Mean WER"],
        [df["cer"].mean() * 100, df["wer"].mean() * 100])
plt.ylabel("Error rate (%)")
plt.title("TrOCR Recognition Error Rates")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_rates_chart.png", dpi=200)
plt.close()

print(f"Images evaluated: {len(df)}")
print(f"Exact match: {accuracy:.2f}%")
print(f"Mean CER: {df['cer'].mean() * 100:.2f}%")
print(f"Mean WER: {df['wer'].mean() * 100:.2f}%")
print("Saved charts:")
print(OUTPUT_DIR / "exact_match_chart.png")
print(OUTPUT_DIR / "error_rates_chart.png")