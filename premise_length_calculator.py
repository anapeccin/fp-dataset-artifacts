from scipy.stats import norm
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to the predictions file
predictions_file = "output/eval_predictions.jsonl"  # Replace with your predictions file path

# Load predictions
with open(predictions_file, "r") as f:
    data = [json.loads(line.strip()) for line in f]

# Compute maximum premise length
max_length = max(len(item["premise"].split()) for item in data)

# Define dynamic premise length bins
num_bins = 6  # Number of bins
bin_size = max_length // num_bins + 1  # Size of each bin
bins = np.arange(0, max_length + bin_size, bin_size)

# Initialize counters
correct_bin_counts = defaultdict(int)
incorrect_bin_counts = defaultdict(int)
total_correct = 0
total_incorrect = 0

# Categorize examples into bins
for item in data:
    premise_length = len(item["premise"].split())  # Use word count for premise length
    bin_index = np.digitize(premise_length, bins) - 1  # Find the appropriate bin

    # Update counts for correct and incorrect examples
    if item["label"] == item["predicted_label"]:
        correct_bin_counts[bin_index] += 1
        total_correct += 1
    else:
        incorrect_bin_counts[bin_index] += 1
        total_incorrect += 1

# Calculate proportions and statistical significance
bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
correct_proportions = []
incorrect_proportions = []
differences = []
p_values = []

for i in range(len(bins) - 1):
    # Counts and totals
    x1 = correct_bin_counts[i]
    n1 = total_correct
    x2 = incorrect_bin_counts[i]
    n2 = total_incorrect

    # Proportions
    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0

    # Difference in proportions
    differences.append(p2 - p1)

    # Save proportions
    correct_proportions.append(p1 * 100)
    incorrect_proportions.append(p2 * 100)

    # Skip bins with no samples in either group
    if n1 == 0 or n2 == 0:
        p_values.append(None)
        continue

    # Pooled proportion
    pooled_p = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))

    # z-score and p-value
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
    p_values.append(p_value)

# Plot the bar chart for proportions
x = np.arange(len(bin_labels))

plt.figure(figsize=(12, 8))
plt.bar(x - 0.2, correct_proportions, width=0.4, label="Correct Distribution", color="blue", alpha=0.7)
plt.bar(x + 0.2, incorrect_proportions, width=0.4, label="Incorrect Distribution", color="red", alpha=0.7)
plt.xticks(x, bin_labels, rotation=45)
plt.xlabel("Premise Length Bins (Word Count)")
plt.ylabel("Percentage of Samples")
plt.title("Correct vs Incorrect Distribution by Premise Length")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the differences in proportions
plt.figure(figsize=(12, 8))
plt.bar(x, differences, width=0.5, color="purple", alpha=0.7)
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.xticks(x, bin_labels, rotation=45)
plt.xlabel("Premise Length Bins (Word Count)")
plt.ylabel("Difference in Proportion")
plt.title("Difference Between Correct and Incorrect Distributions by Premise Length")
plt.tight_layout()
plt.show()

# Display statistical significance results
print("Premise Length Bin\tP-Value\t\tSignificant?")
for i, p_value in enumerate(p_values):
    if p_value is None:
        print(f"{bin_labels[i]}\t\tN/A\t\tN/A")
    else:
        significance = "Yes" if p_value < 0.05 else "No"
        print(f"{bin_labels[i]}\t\t{p_value:.5f}\t\t{significance}")
