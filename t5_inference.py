import torch
import os
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
from helpers import gen_input, verbalise_list, extract_nli_data, get_t5_output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # suppress transformer warnings
warnings.filterwarnings('ignore')  # suppress all other warnings


# 1. model setup
mps_device = torch.device("mps")  # mps = runs on apple silicon gpus instead of just cpus

print("loading model...")
model = T5ForConditionalGeneration.from_pretrained("t5-base", device_map=str("auto")).to(mps_device)

# 2. tokenizer setup
print("\nloading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-base")


# 3. data preparation:
# load 2 datasets (matched + mismatched)
# extract gold_label (gt), sentence 1 (premise), sentence 2 (hypothesis) from datasets
matched_samples = extract_nli_data("dev_matched_sampled-1.jsonl")[:50]
print("matched samples len: ", len(matched_samples))
unmatched_samples = extract_nli_data("dev_mismatched_sampled-1.jsonl")[:50]
print("unmatched samples len: ", len(unmatched_samples))


# 4. task definition
prefix = """
mnli:
Choose one: entailment, contradiction, or neutral.
"""
matched_prompts = []
unmatched_prompts = []

print("\nformat prompts...")
for sample in matched_samples:
    formatted_prompt = gen_input(prompt=prefix, premise=sample[1], hypothesis=sample[2])
    matched_prompts.append(formatted_prompt)

for sample in unmatched_samples:
    formatted_prompt = gen_input(prompt=prefix, premise=sample[1], hypothesis=sample[2])
    unmatched_prompts.append(formatted_prompt)


# divide list into 50 'chunks' each
chunked_matched_prompts = [matched_prompts[i:i + 50] for i in range(0, len(matched_prompts), 50)]
chunked_unmatched_prompts = [unmatched_prompts[i:i + 50] for i in range(0, len(unmatched_prompts), 50)]

print("length of matched chunk: ", len(chunked_matched_prompts[0]), "\nlength of matched chunk list: ", len(chunked_matched_prompts))
print("length of unmatched chunk: ", len(chunked_unmatched_prompts[0]), "\nlength of unmatched chunk list: ", len(chunked_unmatched_prompts))

# 5. input configuration; 6. inference generation: feed input through pipeline + decode output
print("\nmatched samples -- configure inputs; generate and decode outputs...")
total_matched_outputs = []
for chunk in chunked_matched_prompts:
    matched_outputs = get_t5_output(chunk, tokenizer, model, max_length=6)
    total_matched_outputs.extend(matched_outputs)

print("\nunmatched samples -- configure inputs; generate and decode outputs...")
total_unmatched_outputs = []
for chunk in chunked_unmatched_prompts:
    unmatched_outputs = get_t5_output(chunk, tokenizer, model, max_length=6)
    total_unmatched_outputs.extend(unmatched_outputs)

print("\nmatched: ", total_matched_outputs[:10])
print("\nunmatched: ", total_unmatched_outputs[:10])


# verbalise outputs
verbalised_matched_outputs = verbalise_list(total_matched_outputs, "entailment")
verbalised_unmatched_outputs = verbalise_list(total_unmatched_outputs, "entailment")

print("\n[verbalised] first ten matched: ", verbalised_matched_outputs[:10])
print("\n[verbalised] first ten unmatched: ", verbalised_unmatched_outputs[:10])

# 7. evaluation
matched_ground_truths = [sample[0] for sample in matched_samples]
unmatched_ground_truths = [sample[0] for sample in unmatched_samples]

print("matched accuracy: ", accuracy_score(matched_ground_truths, verbalised_matched_outputs))
print("unmatched accuracy: ", accuracy_score(unmatched_ground_truths, verbalised_unmatched_outputs))
