import pandas as pd
from inference.rulebased import calculate_score
from inference.llm import llmApproach
from inference.mlClassifier import mlApproach
from tqdm import tqdm

df = pd.read_csv('training/english.csv')

rule_scores = []
llm_results = []
ml_results = []
combined_results = []

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing prompts"):
    prompt = row['text']

    rule_score = calculate_score(prompt)
    llm_result = llmApproach(prompt)
    ml_result = mlApproach(prompt)

    rule_scores.append(rule_score)
    llm_results.append(llm_result)
    ml_results.append(ml_result)

    if (llm_result or ml_result) and (rule_score > 0):
        combined_results.append(1)
    else:
        combined_results.append(0)


df['rule'] = rule_scores
df['ml'] = ml_results
df['llm'] = llm_results
df['combined'] = combined_results

df.to_csv('test.csv', index=False)

print("Output saved to test.csv")
