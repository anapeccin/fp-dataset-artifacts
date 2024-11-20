import json

# Load the "female-stereo" and "male-stereo" lists from the occupations file
occupations_file_path = 'occup_en-by-type_v1.1.json'
with open(occupations_file_path, 'r') as f:
    occupations_data = json.load(f)

stereo = set(occupations_data["female-stereo"])
# stereo = set(occupations_data["male-stereo"])

# Load the SNLI dataset
snli_file_path = 'snli_clean_validation.jsonl'
output_file_path = 'snli_female_occupation_filter.jsonl'


filtered_occupations = []

with open(snli_file_path, 'r') as f:
    for line in f:
        example = json.loads(line)
        premise = example.get('premise', '').lower()  # Use only "premise" (sentence1)

        # Find matching words from "male-stereo" list in the premise
        matched_words = list(set(premise.split()) & stereo)

        if matched_words:
            # Append the matched words to the example
            example['matched_words'] = matched_words
            filtered_occupations.append(example)

# Save the filtered examples to a new JSONL file
with open(output_file_path, 'w') as f:
    for example in filtered_occupations:
        f.write(json.dumps(example) + '\n')

print(f"Filtered examples with female-stereotypical words saved to: {filtered_occupations}")