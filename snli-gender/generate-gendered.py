import json
import spacy


def edit_snli_file(input_file, output_file):
    """
    Edits the SNLI dataset to replace the subject phrase in the hypothesis
    with 'He', discards plural subjects, and saves to a new file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
    """
    # Load the spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    edited_examples = []

    # Read the input file line by line
    with open(input_file, 'r') as infile:
        for line in infile:
            example = json.loads(line)
            hypothesis = example.get("hypothesis", "")

            # Process the hypothesis with spaCy
            doc = nlp(hypothesis)

            # Find the subject phrase
            subject_token = None
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subject_token = token
                    break

            if subject_token:
                # Discard if subject is plural
                if subject_token.tag_ in ("NNS", "NNPS"):  # Plural nouns
                    continue

                # Replace the subject phrase with "He"
                subject_span = " ".join([word.text for word in subject_token.subtree])
                edited_hypothesis = hypothesis.replace(subject_span, "She", 1)

                # Add the edited example
                example["hypothesis"] = edited_hypothesis
                edited_examples.append(example)

    # Write the edited examples to the output file
    with open(output_file, 'w') as outfile:
        for example in edited_examples:
            outfile.write(json.dumps(example) + '\n')


# Input and output file paths
input_file = "snli_female_occupation_filter.jsonl"
output_file = "snli_female_occupation_filter_hypothesis-she.jsonl"

# Edit the examples
edit_snli_file(input_file, output_file)