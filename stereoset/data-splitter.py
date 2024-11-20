import json
import os

# Paths to input dataset and output directories
input_file = "dev.json"  # Replace with your dataset file path
output_dir = "gender"  # Directory to save the results
os.makedirs(output_dir, exist_ok=True)


# Function to process and split dataset
def split_nested_dataset(input_file, bias_type="gender"):
    # Categories in the dataset
    categories = ["intersentence"]

    # Read the dataset
    with open(input_file, "r") as file:
        nested_data = json.load(file)

    # Ensure the "data" key exists
    if "data" not in nested_data:
        print("Error: 'data' key not found in the dataset.")
        return

    # Process each category (intersentence, intrasentence)
    for category in categories:
        # Prepare category-specific output directory
        # category_dir = os.path.join(output_dir, category)
        # os.makedirs(category_dir, exist_ok=True)

        # Output file paths
        anti_stereotype_file = os.path.join(output_dir, f"{bias_type}_anti_stereotype.jsonl")
        stereotype_file = os.path.join(output_dir, f"{bias_type}_stereotype.jsonl")
        unrelated_file = os.path.join(output_dir, f"{bias_type}_unrelated.jsonl")

        # Open files for writing
        with open(anti_stereotype_file, "w") as anti_file, \
                open(stereotype_file, "w") as stereotype_file, \
                open(unrelated_file, "w") as unrelated_file:

            # Get the list of examples for the current category
            if category in nested_data["data"]:
                examples = nested_data["data"][category]
                for example in examples:
                    if example.get("bias_type") == bias_type:
                        premise = example["context"]  # Rename "context" to "premise"
                        for sentence_data in example.get("sentences", []):
                            hypothesis = sentence_data["sentence"]  # Rename "sentence" to "hypothesis"
                            gold_label = sentence_data["gold_label"]
                            label = 1

                            # Prepare the output object with renamed fields
                            output_object = {"premise": premise, "hypothesis": hypothesis, "label": label}

                            # Write to respective .jsonl file based on the label
                            if gold_label == "anti-stereotype":
                                anti_file.write(json.dumps(output_object) + "\n")
                            elif gold_label == "stereotype":
                                stereotype_file.write(json.dumps(output_object) + "\n")
                            elif gold_label == "unrelated":
                                unrelated_file.write(json.dumps(output_object) + "\n")

        print(f"Processed {category} examples.")
        print(f"Anti-stereotype saved to {anti_stereotype_file}")
        print(f"Stereotype saved to {stereotype_file}")
        print(f"Unrelated saved to {unrelated_file}")


# Call the function
split_nested_dataset(input_file)
