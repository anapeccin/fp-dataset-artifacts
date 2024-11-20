import os
import json
from PyDictionary import PyDictionary
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# Initialize PyDictionary
dictionary = PyDictionary()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract subjects
def extract_subject(sentence):
    """
    Extract the subject of a sentence using spaCy.
    """
    try:
        doc = nlp(sentence)
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "csubj", "agent", "expl"):
                return token.text
        for token in doc:
            if token.pos_ == "NOUN":
                return token.text
        return None
    except Exception as e:
        print(f"Error extracting subject from sentence: {sentence}")
        print(f"Exception: {e}")
        return None
# Function to fetch synonyms using PyDictionary
def get_synonyms_from_pydictionary(word):
    """
    Fetch synonyms for a word using PyDictionary.
    """
    try:
        synonyms = dictionary.synonym(word)
        if synonyms:
            return synonyms
        return None
    except Exception as e:
        print(f"Error fetching synonyms for {word}: {e}")
        return None
# Function to substitute the subject with a PyDictionary synonym
def substitute_subject_with_pydictionary(sentence):
    """
    Replace the subject in a sentence with a PyDictionary synonym.
    """
    subject = extract_subject(sentence)
    if subject:
        synonyms = get_synonyms_from_wordnet(subject)
        if synonyms:
            synonym = synonyms[0]  # Choose the first synonym
            return sentence.replace(subject, synonym, 1), subject, synonym
    return sentence, None, None
# Load JSONL dataset
def load_jsonl(file_path):
    """
    Load a JSON Lines (JSONL) file.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        print(f"Error loading JSONL file: {file_path}")
        print(f"Exception: {e}")
        return []

# Save JSONL file
def save_jsonl(data, file_path):
    """
    Save data to a JSON Lines (JSONL) file, creating directories if necessary.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to file: {file_path}")
        print(f"Exception: {e}")

def get_synonyms_from_wordnet(word):
    """
    Fetch synonyms for a word using NLTK's WordNet.
    """
    try:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():  # Avoid self-substitution
                    synonyms.append(lemma.name().replace('_', ' '))  # Replace underscores with spaces
        return list(set(synonyms))  # Remove duplicates
    except Exception as e:
        print(f"Error fetching synonyms for {word}: {e}")
        return None
# Main processing function
def process_hypotheses_with_pydictionary(input_path, output_path):
    """
    Process hypotheses to replace subjects with synonyms using PyDictionary.
    Save only examples where a substitution occurred.
    """
    data = load_jsonl(input_path)  # Load JSONL dataset
    processed_data = []

    for example in data:
        hypothesis = example["hypothesis"]
        # Substitute the subject in the hypothesis
        new_hypothesis, subject, synonym = substitute_subject_with_pydictionary(hypothesis)
        if subject and synonym:  # Save only if a substitution occurred
            processed_data.append({
                "original_hypothesis": hypothesis,
                "new_hypothesis": new_hypothesis,
                "subject": subject,
                "synonym": synonym,
                "premise": example["premise"],  # Keep the original premise
                "label": example["label"]       # Keep the original label
            })

    # Save processed data
    save_jsonl(processed_data, output_path)
# Main script
if __name__ == "__main__":
    input_path = "snli-hf-original/snli_clean_validation.jsonl"
    output_path = "snli-jitter/snli_clean_validation_subject-synonym.jsonl"

    process_hypotheses_with_pydictionary(input_path, output_path)

