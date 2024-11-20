import json

# Define the input and output file paths
input_file_path = "3-nonstereo_v1.1.json"
output_file_path = "nonstereo.jsonl"

# Process the JSONL file line by line
with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    for line in input_file:
        item = json.loads(line.strip())  # Parse each line as JSON

        # Extract and transform required fields
        filtered_item = {
            "premise": item.pop("sentence1"),
            "hypothesis": item.pop("sentence2"),
            "label": 1 if item["label"] == "neutral" else item["label"]
        }

        # Write filtered item to JSONL
        output_file.write(json.dumps(filtered_item) + "\n")

    print(f"Filtered JSONL file saved at: {output_file_path}")
