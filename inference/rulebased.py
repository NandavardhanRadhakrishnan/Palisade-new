import joblib
from nltk import word_tokenize

# Load the pickled injected data
injected_data = joblib.load('inference\injected_data.pkl')

# Extract the injected verbs, adjectives, and nouns
injected_verbs = injected_data["injected_verbs"]
injected_adjs = injected_data["injected_adjs"]
injected_nouns = injected_data["injected_nouns"]

# Define the calculate_score function


def calculate_score(sentence, n=1):
    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Find the positions of verbs, adjectives, and nouns
    verb_positions = [i for i, word in enumerate(
        words) if word in injected_verbs]
    adj_positions = [i for i, word in enumerate(
        words) if word in injected_adjs]
    noun_positions = [i for i, word in enumerate(
        words) if word in injected_nouns]

    score = 0
    # Check proximity between verbs, adjectives, and nouns
    for v in verb_positions:
        for adj in adj_positions:
            if abs(v - adj) <= n:
                score += 1
        for noun in noun_positions:
            if abs(v - noun) <= n:
                score += 1

    for adj in adj_positions:
        for noun in noun_positions:
            if abs(adj - noun) <= n:
                score += 1

    return score


# # Sample sentence
# sentence = "what is the capital of belgium"

# # Calculate the score for the sentence
# score = calculate_score(sentence, injected_verbs,
#                         injected_adjs, injected_nouns)

# print(f"Score for the sentence: {score}")
