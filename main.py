from inference.rulebased import calculate_score
from inference.llm import llmApproach
from inference.mlClassifier import mlApproach

prompt = input("enter prompt:")

ruleScore = calculate_score(prompt)
llmBool = llmApproach(prompt)
mlBool = mlApproach(prompt)

if (mlBool or llmBool) and (ruleScore > 0):
    print("prompt is injected")
else:
    print("prompt is not injected")
