
import json
import random


def inject_datasets(base_file, inject_file, output_file, inject_ratio):
    """
    Injects examples from `inject_file` into `base_file` at a specified ratio
    and writes the result to `output_file`.

    Parameters:
        base_file (str): Path to the base dataset in JSONL format.
        inject_file (str): Path to the dataset to inject in JSONL format.
        output_file (str): Path to the output JSONL file.
        inject_ratio (float): The ratio of injected samples relative to the base dataset.
    """
    # Read the base dataset
    with open(base_file, 'r') as f:
        base_data = [json.loads(line) for line in f]

    # Read the inject dataset
    with open(inject_file, 'r') as f:
        inject_data = [json.loads(line) for line in f]

    # Calculate the number of samples to inject
    num_to_inject = int(len(base_data) * inject_ratio)

    # Shuffle the inject dataset to ensure random sampling
    random.shuffle(inject_data)

    # Select the first `num_to_inject` samples from the shuffled inject dataset
    selected_inject_data = inject_data[:num_to_inject]

    # Combine the datasets
    combined_data = base_data + selected_inject_data

    # Shuffle the combined dataset to mix injected data with the base data
    random.shuffle(combined_data)

    # Write the combined dataset to the output file
    with open(output_file, 'w') as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')


# Parameters
base_file_path = "snli-hf-original/snli_clean_train.jsonl"
inject_file_path = "snli-gender/train/snli_gender_occupation_filter_antistereo-gendered-hypothesis.jsonl"
output_file_path = "snli-gender/train/snli-hf-original-plus-antistereo-0.2ratio.jsonl"
injection_ratio = 0.2  # 20% of the base dataset size

# Inject datasets
inject_datasets(base_file_path, inject_file_path, output_file_path, injection_ratio)
