import json
from collections import Counter, defaultdict

# Path to the saved predictions (JSON file)
output_file = "output/eval_predictions.jsonl"  # Adjust the file path as needed

# Load the predictions
with open(output_file, "r") as f:
    data = [json.loads(line.strip()) for line in f]

# Initialize counters
incorrect_by_true_label = defaultdict(Counter)
predicted_label_totals = Counter()

# Process predictions
for item in data:
    label = item["label"]
    predicted_label = item["predicted_label"]

    if label != predicted_label:
        # Count incorrect predictions by true label
        incorrect_by_true_label[label][predicted_label] += 1
        # Count predicted labels for all incorrect predictions
        predicted_label_totals[predicted_label] += 1

# Output incorrect predictions breakdown by true label
print("Incorrect Predictions Breakdown by True Label:")
for true_label, pred_counter in incorrect_by_true_label.items():
    total_incorrect = sum(pred_counter.values())
    print(f"\nTrue Label {true_label}: {total_incorrect} total")
    for predicted_label, count in pred_counter.items():
        percentage = (count / total_incorrect) * 100
        print(f"  Predicted as {predicted_label}: {count} ({percentage:.2f}%)")

# Output breakdown of predicted labels across all incorrect predictions
print("\nOverall Breakdown of Predicted Labels for Incorrect Predictions:")
total_incorrect_predictions = sum(predicted_label_totals.values())
for predicted_label, count in predicted_label_totals.items():
    percentage = (count / total_incorrect_predictions) * 100
    print(f"  Predicted Label {predicted_label}: {count} ({percentage:.2f}%)")

# Initialize counters
true_label_by_predicted = defaultdict(Counter)

# Process predictions
for item in data:
    label = item["label"]
    predicted_label = item["predicted_label"]

    if label != predicted_label:
        # Count true labels for each incorrect predicted label
        true_label_by_predicted[predicted_label][label] += 1

# Output breakdown of true labels for each incorrectly predicted label
print("Incorrect Predictions Breakdown by Predicted Label:")
for predicted_label, true_counter in true_label_by_predicted.items():
    total_for_predicted = sum(true_counter.values())
    print(f"\nPredicted Label {predicted_label}: {total_for_predicted} total")
    for true_label, count in true_counter.items():
        percentage = (count / total_for_predicted) * 100
        print(f"  True Label {true_label}: {count} ({percentage:.2f}%)")