import pandas as pd
import spacy
import joblib
from nltk import word_tokenize
import re
# Load the dataset
df = pd.read_csv('english.csv')

# Normalize text


def normalizeText(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]|_", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


df['text'] = df['text'].apply(normalizeText)

# Filter out rows with the label == 1
injected = df[df['label'] == 1]

# Load Spacy model
nlp = spacy.load("en_core_web_md")

# Function to filter words based on similarity


def filter_similar_words(input_words, target_tokens, threshold=0.5):
    similar_words = []
    for word in input_words:
        token = nlp(word)
        max_similarity = max([token.similarity(target)
                             for target in target_tokens])
        if max_similarity >= threshold:
            similar_words.append(word)
    return similar_words

# Function to extract tokens of a particular POS type


def get_tokens(sentence, token_type):
    doc = nlp(sentence)
    words = [token.text for token in doc if token.pos_ == token_type]
    return words


# Tokenize the text
injected['text'] = injected['text'].apply(word_tokenize)
injected['text'] = injected['text'].apply(" ".join)

# Extract verbs, adjectives, and nouns from the injected texts
verbs = [word for sent in injected['text']
         for word in get_tokens(sent, "VERB")]
adjectives = [word for sent in injected['text']
              for word in get_tokens(sent, "ADJ")]
nouns = [word for sent in injected['text']
         for word in get_tokens(sent, "NOUN")]

# Target verbs, adjectives, and nouns
target_verbs = [nlp("ignore"), nlp("forget")]
target_adjs = [nlp("previous"), nlp("above")]
target_nouns = [nlp("content"), nlp("input"), nlp("text")]

# Filter injected verbs, adjectives, and nouns based on similarity
injected_verbs = filter_similar_words(verbs, target_verbs, threshold=0.3)
injected_adjs = filter_similar_words(adjectives, target_adjs, threshold=0.3)
injected_nouns = filter_similar_words(nouns, target_nouns, threshold=0.3)

# Display the filtered sets (just for sanity check)
print(f"Injected Verbs: {set(injected_verbs)}")
print(f"Injected Adjectives: {set(injected_adjs)}")
print(f"Injected Nouns: {set(injected_nouns)}")

# Save the injected data (verbs, adjectives, nouns) using joblib
injected_data = {
    "injected_verbs": injected_verbs,
    "injected_adjs": injected_adjs,
    "injected_nouns": injected_nouns,
}

# Pickle the data for future inference
joblib.dump(injected_data, 'injected_data.pkl')

print("Injected data saved to injected_data.pkl")
