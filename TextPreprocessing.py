import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Text preprocessing function
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Apply NLP pipeline
    doc = nlp(text)
    # Extract entities and lemmatize
    simplified_text = " ".join([token.lemma_ for token in doc])
    return simplified_text

# Example usage
example_text = "The patient has a glioblastoma in the frontal lobe, confirmed by MRI."
cleaned_text = preprocess_text(example_text)
print("Original Text:", example_text)
print("Processed Text:", cleaned_text)
