import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


# load model 
print("load model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print("\nload tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# load 2 datasets (matched + mismatched)
# extract gold_label (gt), sentence 1 (premise), sentence 2 (hypothesis) from datasets
print("\nextracting data...")
matched_samples = []
with open("dev_matched_sampled-1.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)
        matched_samples.append([json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]])


unmatched_samples = []
with open("dev_mismatched_sampled-1.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)
        unmatched_samples.append([json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]])


# prompt for gpt2:
prompt = """
I am going to show you two sentences: a premise and a hypothesis. I'll explain the relationship between them.
The relationship can be one of the following:
- Entailment: The hypothesis logically follows from the premise.
- Contradiction: The hypothesis logically contradicts the premise.
- Neutral: The hypothesis neither follows nor contradicts the premise.

Here are the sentences with their relationship:
"""

# format the prompt with the correct 'premise' and 'hypothesis'
def gen_input(line: list[str, str, str]) -> str:
    premise = line[1]
    hypothesis = line[2]
    formatted_prompt = prompt + "\nPremise: " + premise + "\nHypothesis: " + hypothesis + "\nResult: "
    return formatted_prompt


# set model w/ appropriate decoding hyperparams (low max length, 0.7 temperature, etc)
def gen_output(prompt: str) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
    input_ids,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=4)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


matched_results = []
unmatched_results = []


print("\ngenerating outputs...")
# generate results for each matched/unmatched sample
for sample in matched_samples:
    formatted_prompt = gen_input(sample)
    output = gen_output(formatted_prompt)
    cleaned_output = output.split("Result: ", 1)[1]  # split into prompt ([0]) and output ([1]), get just output
    matched_results.append(cleaned_output)

for sample in unmatched_samples:
    formatted_prompt = gen_input(sample)
    output = gen_output(formatted_prompt)
    cleaned_output = output.split("Result: ", 1)[1]  # split into prompt ([0]) and output ([1]), get just output
    unmatched_results.append(cleaned_output)


# create verbaliser (converting output and such)
def verbalise(output: str) -> str:
    output = output.lower()
    if("entail" in output or "entailment" in output or "true" in output or "correct" in output):
        return "entailment"
    elif("contradicts" in output or "contradiction" in output or "false" in output or "wrong" in output):
        return "contradiction"
    elif("neutral" in output or "ambiguous" in output):
        return "neutral"
    else:
        return "no meaningful result"


print("\nverbalising results...")
# match verbalised result to its original sample:
for i in range(len(matched_results)):
    output = matched_results[i]
    verbalised_output = verbalise(output)
    matched_samples[i].append(verbalised_output)

for i in range(len(unmatched_results)):
    output = unmatched_results[i]
    verbalised_output = verbalise(output)
    unmatched_samples[i].append(verbalised_output)

print("\nevaluating results...")
# evaluate -- get accuracy, f1 score
matched_ground_truths = [sample[0] for sample in matched_samples]
unmatched_ground_truths = [sample[0] for sample in unmatched_samples]


matched_predictions = [sample[3] for sample in matched_samples]
unmatched_predictions = [sample[3] for sample in unmatched_samples]


print("matched accuracy: ", accuracy_score(matched_ground_truths, matched_predictions))
print("unmatched accuracy: ", accuracy_score(unmatched_ground_truths, unmatched_predictions))
