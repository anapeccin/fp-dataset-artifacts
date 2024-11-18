import pandas as pd
import json

# Load the .tsv file into a pandas DataFrame
input_file = "snli-lit/original+basic/dev.tsv"  # Replace with the actual path to your file
output_file = "snli-lit/dev-original-onlyperturbed.jsonl"       # Replace with your desired output file name


# Load the TSV file
df = pd.read_csv(input_file, sep="\t")

# Identify indexes that appear more than once
repeated_indexes = df['index'].value_counts()
repeated_indexes = repeated_indexes[repeated_indexes > 1].index

# Filter the dataset for rows with repeated indexes and captionID as "original"
filtered_df = df[(df['index'].isin(repeated_indexes)) & (df['captionID'] == "original")]

# Select relevant columns
filtered_df = filtered_df[['sentence1', 'sentence2', 'gold_label']]

# Map labels to numerical values
label_mapping = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}
filtered_df['label'] = filtered_df['gold_label'].map(label_mapping)

# Drop rows with missing or invalid labels
filtered_df = filtered_df.dropna(subset=['label'])

# Prepare the dataset in the desired format
data = []
for _, row in filtered_df.iterrows():
    entry = {
        "premise": row['sentence1'],
        "hypothesis": row['sentence2'],
        "label": int(row['label'])
    }
    data.append(entry)

# Save the dataset in JSONL format
with open(output_file, "w", encoding="utf-8") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

print(f"Filtered dataset saved to {output_file}")