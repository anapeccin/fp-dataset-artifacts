import json

# Path to the input JSON file
input_file = "output-snli-lit-dev-perturbed-baseline/eval_predictions.jsonl"

# Path to save filtered mismatches
output_file = "output-snli-lit-dev-perturbed-baseline/wrong_predictions.json"

# Read the JSON file line by line and filter mismatches
mismatches = []
with open(input_file, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        if data["label"] != data["predicted_label"]:
            mismatches.append(data)

# Save mismatches to a new JSON file
with open(output_file, "w") as f:
    for mismatch in mismatches:
        f.write(json.dumps(mismatch) + "\n")

print(f"Filtered mismatches saved to {output_file}")