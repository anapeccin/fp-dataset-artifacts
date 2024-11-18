import json
from collections import defaultdict
import numpy as np
from scipy.stats import norm

import pandas as pd
from joypy import joyplot
import matplotlib.pyplot as plt

# Path to the predictions file
predictions_file = "output/eval_predictions.jsonl"  # Replace with your predictions file path

# Function to calculate lexical overlap
# def compute_lexical_overlap(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Group lexical overlaps for incorrect predictions by true and predicted class
# incorrect_by_true_and_predicted_class = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list)}
#
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     if item["label"] != item["predicted_label"]:
#         if item["label"] in incorrect_by_true_and_predicted_class:
#             incorrect_by_true_and_predicted_class[item["label"]][item["predicted_label"]].append(overlap)
#
# # Prepare and plot distributions for each true class
# for true_class, predicted_classes in incorrect_by_true_and_predicted_class.items():
#     # Prepare data for ridgeline plot
#     df = pd.concat(
#         [pd.DataFrame({"Overlap": overlaps, "Predicted Class": [pred_class] * len(overlaps)})
#          for pred_class, overlaps in predicted_classes.items()]
#     )
#
#     # Plot ridgeline for each true class with overlay of predicted classes
#     plt.figure(figsize=(12, 8))
#     joyplot(
#         data=df,
#         by="Predicted Class",
#         column="Overlap",
#         figsize=(12, 8),
#         grid="y",
#         title=f"Lexical Overlap Distributions for True Class {true_class} by Predicted Class"
#     )
#     plt.xlabel("Lexical Overlap")
#     plt.show()
#
# # Function to calculate lexical overlap
# def compute_binned_lexical_overlap(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
#     # Load predictions
#
#
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
#     # Define bins and initialize counters
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
# correct_bin_counts = defaultdict(int)
# incorrect_bin_counts = defaultdict(int)
#
# total_correct = 0
# total_incorrect = 0
#
# # Compute lexical overlaps and categorize as correct or incorrect
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#
#     if item["label"] == item["predicted_label"]:
#         correct_bin_counts[bin_index] += 1
#         total_correct += 1
#     else:
#         incorrect_bin_counts[bin_index] += 1
#         total_incorrect += 1
#
# # Calculate percentages for each bin
# bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
# correct_percentages = [
#     (correct_bin_counts[i] / total_correct * 100 if total_correct > 0 else 0)
#     for i in range(len(bins) - 1)
# ]
# incorrect_percentages = [
#     (incorrect_bin_counts[i] / total_incorrect * 100 if total_incorrect > 0 else 0)
#     for i in range(len(bins) - 1)
# ]
#
# # Plot the bar chart
# x = np.arange(len(bin_labels))  # Bin indices
# width = 0.4  # Width of the bars
#
# plt.figure(figsize=(12, 8))
# plt.bar(x - width / 2, correct_percentages, width, label="Correct", color="blue", alpha=0.7)
# plt.bar(x + width / 2, incorrect_percentages, width, label="Incorrect", color="red", alpha=0.7)
# plt.xticks(x, bin_labels, rotation=45)
# plt.xlabel("Lexical Overlap Bins")
# plt.ylabel("Percentage of Samples")
# plt.title("Distribution of Correct vs Incorrect Predictions Across Lexical Overlap Bins")
# plt.legend()
# plt.tight_layout()
# plt.show()

# def compute_lexical_overlap_true_class(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Define bins
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
#
# # Initialize counters for each true class label
# true_class_bin_counts = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}
# true_class_totals = defaultdict(int)
#
# # Compute lexical overlaps and group by true class label
# for item in data:
#     overlap = compute_lexical_overlap_true_class(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#     true_label = item["label"]
#
#     # Increment bin counts for the true label
#     if true_label in true_class_bin_counts:
#         true_class_bin_counts[true_label][bin_index] += 1
#         true_class_totals[true_label] += 1
#
# # Calculate percentages for each true label and bin
# bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
# percentages = {
#     label: [
#         (true_class_bin_counts[label][i] / true_class_totals[label] * 100 if true_class_totals[label] > 0 else 0)
#         for i in range(len(bins) - 1)
#     ]
#     for label in true_class_bin_counts
# }
#
# # Plot the bar chart
# x = np.arange(len(bin_labels))  # Bin indices
# width = 0.2  # Width of the bars
# colors = ["blue", "orange", "green"]
#
# plt.figure(figsize=(12, 8))
#
# for i, (label, color) in enumerate(zip(true_class_bin_counts.keys(), colors)):
#     plt.bar(x + i * width, percentages[label], width, label=f"Label {label}", color=color, alpha=0.7)
#
# plt.xticks(x + width, bin_labels, rotation=45)
# plt.xlabel("Lexical Overlap Bins")
# plt.ylabel("Percentage of Samples")
# plt.title("Lexical Overlap Distribution Within Each True Class Label")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Function to calculate lexical overlap
# def compute_lexical_overlap(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Define bins
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
#
# # Initialize counters for incorrectly classified examples
# incorrect_bin_counts = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}
# incorrect_totals = defaultdict(int)
#
# # Compute lexical overlaps and group by true class label for incorrect predictions
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#     true_label = item["label"]
#
#     # Increment counts only for incorrectly classified examples
#     if item["label"] != item["predicted_label"]:
#         if true_label in incorrect_bin_counts:
#             incorrect_bin_counts[true_label][bin_index] += 1
#             incorrect_totals[true_label] += 1
#
# # Calculate percentages for each true label and bin
# bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
# percentages = {
#     label: [
#         (incorrect_bin_counts[label][i] / incorrect_totals[label] * 100 if incorrect_totals[label] > 0 else 0)
#         for i in range(len(bins) - 1)
#     ]
#     for label in incorrect_bin_counts
# }
#
# # Plot the bar chart
# x = np.arange(len(bin_labels))  # Bin indices
# width = 0.2  # Width of the bars
# colors = ["blue", "orange", "green"]
#
# plt.figure(figsize=(12, 8))
#
# for i, (label, color) in enumerate(zip(incorrect_bin_counts.keys(), colors)):
#     plt.bar(x + i * width, percentages[label], width, label=f"True Label {label}", color=color, alpha=0.7)
#
# plt.xticks(x + width, bin_labels, rotation=45)
# plt.xlabel("Lexical Overlap Bins")
# plt.ylabel("Percentage of Incorrectly Classified Samples")
# plt.title("Lexical Overlap Distribution for Incorrect Predictions by True Label")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Function to calculate lexical overlap
# def compute_lexical_overlap(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Define bins
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
#
# # Initialize counters for incorrectly classified examples by true and predicted labels
# incorrect_by_true_and_predicted = {0: defaultdict(lambda: defaultdict(int)),
#                                    1: defaultdict(lambda: defaultdict(int)),
#                                    2: defaultdict(lambda: defaultdict(int))}
# totals_by_true_and_predicted = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}
#
# # Compute lexical overlaps and group by true and predicted labels for incorrect predictions
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#     true_label = item["label"]
#     predicted_label = item["predicted_label"]
#
#     # Increment counts only for incorrectly classified examples
#     if true_label != predicted_label:
#         incorrect_by_true_and_predicted[true_label][predicted_label][bin_index] += 1
#         totals_by_true_and_predicted[true_label][predicted_label] += 1
#
# # Prepare data for plotting
# for true_label in incorrect_by_true_and_predicted:
#     bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
#     percentages_by_predicted = {
#         predicted_label: [
#             (incorrect_by_true_and_predicted[true_label][predicted_label][i] /
#              totals_by_true_and_predicted[true_label][predicted_label] * 100
#              if totals_by_true_and_predicted[true_label][predicted_label] > 0 else 0)
#             for i in range(len(bins) - 1)
#         ]
#         for predicted_label in incorrect_by_true_and_predicted[true_label]
#     }
#
#     # Plot the bar chart
#     x = np.arange(len(bin_labels))  # Bin indices
#     width = 0.2  # Width of the bars
#
#     plt.figure(figsize=(12, 8))
#     for i, (predicted_label, percentages) in enumerate(percentages_by_predicted.items()):
#         plt.bar(x + i * width, percentages, width, label=f"Predicted as {predicted_label}")
#
#     plt.xticks(x + width, bin_labels, rotation=45)
#     plt.xlabel("Lexical Overlap Bins")
#     plt.ylabel("Percentage of Samples")
#     plt.title(f"Lexical Overlap Distribution for True Label {true_label} by Predicted Label")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Function to calculate lexical overlap
# def compute_lexical_overlap(premise, hypothesis):
#     """
#     Compute lexical overlap as the ratio of shared words to total unique words.
#     Args:
#         premise (str): The premise sentence.
#         hypothesis (str): The hypothesis sentence.
#     Returns:
#         float: Lexical overlap ratio.
#     """
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Define bins
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
#
# # Initialize counters for lexical distributions
# true_class_bin_counts = defaultdict(lambda: defaultdict(int))  # Overall true class distribution
# totals_true_class = defaultdict(int)
#
# incorrect_by_true_and_predicted = {0: defaultdict(lambda: defaultdict(int)),
#                                    1: defaultdict(lambda: defaultdict(int)),
#                                    2: defaultdict(lambda: defaultdict(int))}
# totals_by_true_and_predicted = {0: defaultdict(int), 1: defaultdict(int), 2: defaultdict(int)}
#
# # Compute lexical overlaps and group by true and predicted labels
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#     true_label = item["label"]
#     predicted_label = item["predicted_label"]
#
#     # Update true class distribution
#     true_class_bin_counts[true_label][bin_index] += 1
#     totals_true_class[true_label] += 1
#
#     # Update incorrect predictions distribution
#     if true_label != predicted_label:
#         incorrect_by_true_and_predicted[true_label][predicted_label][bin_index] += 1
#         totals_by_true_and_predicted[true_label][predicted_label] += 1
#
# # Prepare data for plotting
# for true_label in incorrect_by_true_and_predicted:
#     bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
#
#     # Calculate percentages for incorrectly predicted labels
#     percentages_by_predicted = {
#         predicted_label: [
#             (incorrect_by_true_and_predicted[true_label][predicted_label][i] /
#              totals_by_true_and_predicted[true_label][predicted_label] * 100
#              if totals_by_true_and_predicted[true_label][predicted_label] > 0 else 0)
#             for i in range(len(bins) - 1)
#         ]
#         for predicted_label in incorrect_by_true_and_predicted[true_label]
#     }
#
#     # Calculate percentages for true class distribution
#     true_class_percentages = [
#         (true_class_bin_counts[true_label][i] / totals_true_class[true_label] * 100
#          if totals_true_class[true_label] > 0 else 0)
#         for i in range(len(bins) - 1)
#     ]
#
#     # Plot the bar chart
#     x = np.arange(len(bin_labels))  # Bin indices
#     width = 0.2  # Width of the bars
#
#     plt.figure(figsize=(12, 8))
#
#     # Plot true class distribution
#     plt.bar(x - width, true_class_percentages, width, label=f"True Class {true_label}", color="gray", alpha=0.6)
#
#     # Plot incorrectly predicted label distributions
#     for i, (predicted_label, percentages) in enumerate(percentages_by_predicted.items()):
#         plt.bar(x + i * width, percentages, width, label=f"Predicted as {predicted_label}")
#
#     plt.xticks(x, bin_labels, rotation=45)
#     plt.xlabel("Lexical Overlap Bins")
#     plt.ylabel("Percentage of Samples")
#     plt.title(f"Lexical Overlap Distribution for True Label {true_label} with Incorrect Predictions")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Function to calculate lexical overlap
# def compute_lexical_overlap(premise, hypothesis):
#     premise_tokens = set(premise.lower().split())
#     hypothesis_tokens = set(hypothesis.lower().split())
#     intersection = premise_tokens & hypothesis_tokens
#     union = premise_tokens | hypothesis_tokens
#     return len(intersection) / len(union) if union else 0
#
# # Load predictions
# with open(predictions_file, "r") as f:
#     data = [json.loads(line.strip()) for line in f]
#
# # Define bins
# bins = np.linspace(0, 1, 11)  # Define 10 bins (0.0 to 1.0)
#
# # Initialize counters
# true_class_bin_counts = defaultdict(lambda: defaultdict(int))  # Overall true class distribution
# incorrect_bin_counts = defaultdict(lambda: defaultdict(int))  # Incorrect predictions distribution
# totals_true_class = defaultdict(int)
# totals_incorrect = defaultdict(int)
#
# # Compute lexical overlaps and group by true labels
# for item in data:
#     overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
#     bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin
#     true_label = item["label"]
#     predicted_label = item["predicted_label"]
#
#     # Update true class distribution
#     true_class_bin_counts[true_label][bin_index] += 1
#     totals_true_class[true_label] += 1
#
#     # Update incorrect predictions distribution
#     if true_label != predicted_label:
#         incorrect_bin_counts[true_label][bin_index] += 1
#         totals_incorrect[true_label] += 1
#
# # Calculate proportions and differences
# bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
# differences_by_true_label = {}
#
# for true_label in true_class_bin_counts:
#     true_proportions = [
#         (true_class_bin_counts[true_label][i] / totals_true_class[true_label]
#          if totals_true_class[true_label] > 0 else 0)
#         for i in range(len(bins) - 1)
#     ]
#     incorrect_proportions = [
#         (incorrect_bin_counts[true_label][i] / totals_incorrect[true_label]
#          if totals_incorrect[true_label] > 0 else 0)
#         for i in range(len(bins) - 1)
#     ]
#     differences_by_true_label[true_label] = [
#         incorrect_proportions[i] - true_proportions[i] for i in range(len(bins) - 1)
#     ]
#
#     # Plot the differences for each true label
#     plt.figure(figsize=(12, 8))
#     x = np.arange(len(bin_labels))
#     plt.bar(x, differences_by_true_label[true_label], color="purple", alpha=0.7)
#     plt.axhline(0, color="black", linewidth=1, linestyle="--")
#     plt.xticks(x, bin_labels, rotation=45)
#     plt.xlabel("Lexical Overlap Bins")
#     plt.ylabel("Difference in Proportion")
#     plt.title(f"Difference Between True and Incorrect Distributions for True Label {true_label}")
#     plt.tight_layout()
#     plt.show()
#
#
# Function to calculate lexical overlap
# Function to calculate lexical overlap
def compute_lexical_overlap(premise, hypothesis):
    premise_tokens = set(premise.lower().split())
    hypothesis_tokens = set(hypothesis.lower().split())
    intersection = premise_tokens & hypothesis_tokens
    union = premise_tokens | hypothesis_tokens
    return len(intersection) / len(union) if union else 0

# Load predictions
with open(predictions_file, "r") as f:
    data = [json.loads(line.strip()) for line in f]

# Define bins (expanded to 0.2 intervals)
bins = np.linspace(0, 1, 6)  # Define bins: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Initialize counters for true and incorrect examples
true_bin_counts = defaultdict(int)
incorrect_bin_counts = defaultdict(int)
total_true = 0
total_incorrect = 0

# Compute lexical overlaps and group overall
for item in data:
    overlap = compute_lexical_overlap(item["premise"], item["hypothesis"])
    bin_index = np.digitize(overlap, bins) - 1  # Find the appropriate bin

    # Update true distribution
    true_bin_counts[bin_index] += 1
    total_true += 1

    # Update incorrect distribution if misclassified
    if item["label"] != item["predicted_label"]:
        incorrect_bin_counts[bin_index] += 1
        total_incorrect += 1

# Calculate proportions
true_proportions = [
    (true_bin_counts[i] / total_true * 100 if total_true > 0 else 0)
    for i in range(len(bins) - 1)
]
incorrect_proportions = [
    (incorrect_bin_counts[i] / total_incorrect * 100 if total_incorrect > 0 else 0)
    for i in range(len(bins) - 1)
]

# Calculate the difference between incorrect and true proportions
differences = [incorrect_proportions[i] - true_proportions[i] for i in range(len(bins) - 1)]

# Plot the results
bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

# Plot true and incorrect distributions
plt.figure(figsize=(12, 8))
x = np.arange(len(bin_labels))
plt.bar(x - 0.2, true_proportions, width=0.4, label="True Distribution", color="blue", alpha=0.7)
plt.bar(x + 0.2, incorrect_proportions, width=0.4, label="Incorrect Distribution", color="red", alpha=0.7)
plt.xticks(x, bin_labels, rotation=45)
plt.xlabel("Lexical Overlap Bins")
plt.ylabel("Percentage of Samples")
plt.title("True vs Incorrect Distribution (Expanded Bins: 0.2 Interval)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the differences
plt.figure(figsize=(12, 8))
plt.bar(x, differences, width=0.5, color="purple", alpha=0.7)
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.xticks(x, bin_labels, rotation=45)
plt.xlabel("Lexical Overlap Bins")
plt.ylabel("Difference in Proportion")
plt.title("Difference Between True and Incorrect Distributions (Expanded Bins: 0.2 Interval)")
plt.tight_layout()
plt.show()

# Calculate proportions and significance for each bin
bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
significance_results = []

for i in range(len(bins) - 1):
    # Counts and totals for true and incorrect distributions
    x1 = true_bin_counts[i]
    n1 = total_true
    x2 = incorrect_bin_counts[i]
    n2 = total_incorrect

    # Skip bins where either group has no examples
    if n1 == 0 or n2 == 0:
        significance_results.append((bin_labels[i], None))
        continue

    # Proportions
    p1 = x1 / n1
    p2 = x2 / n2

    # Pooled proportion
    p = (x1 + x2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    # z-score
    z = (p1 - p2) / se

    # p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test

    significance_results.append((bin_labels[i], p_value))

# Print significance results
print("Bin\t\tP-Value\t\tSignificant?")
for bin_label, p_value in significance_results:
    if p_value is None:
        print(f"{bin_label}\t\tN/A\t\tN/A")
    else:
        print(f"{bin_label}\t\t{p_value:.5f}\t\t{'Yes' if p_value < 0.05 else 'No'}")