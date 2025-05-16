import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
from helpers import gen_input, extract_nli_data, gen_gpt2_output, verbalise_list
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


# load model 
print("load model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print("\nload tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# load 2 datasets (matched + mismatched)
# extract gold_label (gt), sentence 1 (premise), sentence 2 (hypothesis) from datasets

matched_samples = extract_nli_data("dev_matched_sampled-1.jsonl")
print("matched samples len: ", len(matched_samples))
unmatched_samples = extract_nli_data("dev_mismatched_sampled-1.jsonl")
print("unmatched samples len: ", len(unmatched_samples))


# prompt:
prompt = """
I am going to show you two sentences: a premise and a hypothesis. I'll explain the relationship between them.
The relationship can be one of the following:
- Entailment: The hypothesis logically follows from the premise.
- Contradiction: The hypothesis logically contradicts the premise.
- Neutral: The hypothesis neither follows nor contradicts the premise.

Premise: My friend Frank went to the supermarket earlier today. Hypothesis: Frank didn't go yet, so he'll drive to the supermarket later today.
Relationship: Contradiction

Premise: She would never know what happened. Hypothesis: And she never was able to find out what occurred.
Relationship: Entailment



"""


print("\ngenerating outputs...")

# generate results for each matched/unmatched sample
matched_results = []
unmatched_results = []

for sample in matched_samples:
    formatted_prompt = gen_input(prompt=prompt, premise=sample[1], hypothesis=sample[2])
    output = gen_gpt2_output(prompt=formatted_prompt, model=model, tokenizer=tokenizer, max_tokens=10)
    cleaned_output = output.split("Relationship: ", 3)[3]  # split into prompt ([0]) and output ([1]), get just output
    matched_results.append(cleaned_output)

for sample in unmatched_samples:
    formatted_prompt = gen_input(prompt=prompt, premise=sample[1], hypothesis=sample[2])
    output = gen_gpt2_output(prompt=formatted_prompt, model=model, tokenizer=tokenizer, max_tokens=10)
    cleaned_output = output.split("Relationship: ", 3)[3]  # split into prompt ([0]) and output ([1]), get just output
    unmatched_results.append(cleaned_output)

print("results: ", matched_results[:10])
# verbalise results
print("\nverbalising results...")
verbalised_matched_results = verbalise_list(matched_results, "entailment")
verbalised_unmatched_results = verbalise_list(unmatched_results, "entailment")

print("verbalised results: ", verbalised_matched_results[:10])


# evaluate -- get accuracy
print("\nevaluating results...")
matched_ground_truths = [sample[0] for sample in matched_samples]
unmatched_ground_truths = [sample[0] for sample in unmatched_samples]

print("matched accuracy: ", accuracy_score(matched_ground_truths, verbalised_matched_results))
print("unmatched accuracy: ", accuracy_score(unmatched_ground_truths, verbalised_unmatched_results))
