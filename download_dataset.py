from datasets import load_dataset

# Load the SNLI dataset
dataset = load_dataset("snli")

# Define valid labels
valid_labels = [0, 1, 2]  # 0: entailment, 1: neutral, 2: contradiction

# Remove examples with missing or invalid labels
def filter_invalid_labels(example):
    return example['label'] in valid_labels

filtered_dataset = dataset.filter(filter_invalid_labels)

# Save the cleaned dataset to JSON
filtered_dataset["train"].to_json("snli-hf-original/snli_clean_train.jsonl")
filtered_dataset["validation"].to_json("snli-hf-original/snli_clean_validation.jsonl")
filtered_dataset["test"].to_json("snli-hf-original/snli_clean_test.jsonl")

print("Cleaned dataset saved!")