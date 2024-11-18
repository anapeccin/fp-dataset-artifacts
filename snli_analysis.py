from datasets import load_dataset

# Load the SNLI dataset
dataset = load_dataset("snli")

# Function to calculate label percentages
def calculate_label_percentages(data_split):
    """
    Calculate the percentages of each label in a dataset split.
    Args:
        data_split: A Hugging Face Dataset split (e.g., train, test, validation).
    Returns:
        A dictionary with percentages of labels 0, 1, and 2.
    """
    total = len(data_split["label"])
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in data_split["label"]:
        if label in label_counts:
            label_counts[label] += 1
    return {label: (count / total) * 100 for label, count in label_counts.items()}

# Analyze train, validation, and test sets
for split_name in ["train", "validation", "test"]:
    split = dataset[split_name]
    percentages = calculate_label_percentages(split)
    print(f"Label distribution for {split_name} set (percentages):")
    print(f"  Entailment (0): {percentages[0]:.2f}%")
    print(f"  Neutral (1): {percentages[1]:.2f}%")
    print(f"  Contradiction (2): {percentages[2]:.2f}%")
    print()
