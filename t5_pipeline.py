import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
import datetime
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


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
def extract_data(json_name: str) -> list:
    print("\nextracting data...")
    samples = []
    with open(json_name, "r") as file:
        for line in file:
            json_line = json.loads(line)
            samples.append([json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]])
    return samples

matched_samples = extract_data("dev_matched_sampled-1.jsonl")
print("matched samples len: ", len(matched_samples))

unmatched_samples = extract_data("dev_mismatched_sampled-1.jsonl")
print("unmatched samples len: ", len(unmatched_samples))


# 4. task definition
prefix = """
mnli:
Choose one: entailment, contradiction, or neutral.
"""

# format the prompt with the correct 'premise' and 'hypothesis'
def gen_input(line: list[str, str, str]) -> str:
    premise = line[1]
    hypothesis = line[2]
    completed_prompt = prefix + " Premise: " + premise + " Hypothesis: " + hypothesis + " Result: "
    return completed_prompt

matched_prompts = []
unmatched_prompts = []

print("\nformat prompts...")
for sample in matched_samples:
    formatted_prompt = gen_input(sample)
    matched_prompts.append(formatted_prompt)

for sample in unmatched_samples:
    formatted_prompt = gen_input(sample)
    unmatched_prompts.append(formatted_prompt)


# divide list into 50 'chunks' each
chunked_matched_prompts = [matched_prompts[i:i + 50] for i in range(0, len(matched_prompts), 50)]
chunked_unmatched_prompts = [unmatched_prompts[i:i + 50] for i in range(0, len(unmatched_prompts), 50)]

print("length of matched chunk: ", len(chunked_matched_prompts[0]), "\nlength of matched chunk list: ", len(chunked_matched_prompts))
print("length of unmatched chunk: ", len(chunked_unmatched_prompts[0]), "\nlength of unmatched chunk list: ", len(chunked_unmatched_prompts))


# 5. input configuration
def tokenize_inputs(prompts: list):
    tokenized_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"].to("mps")  # move tensors to MPS
    attention_mask = tokenized_inputs["attention_mask"].to("mps")
    return [tokenized_inputs, input_ids, attention_mask]


# 6. inference generation: feed input through pipeline + decode output
def gen_output(input_ids, attention_mask, max_length) -> list:
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded_output

print("\nmatched samples -- configure inputs; generate and decode outputs...")
start_time = datetime.now()
total_decoded_matched_outputs = []
for chunk in chunked_matched_prompts:
    [tokenized_inputs_matched, input_ids_matched, attention_mask_matched] = tokenize_inputs(chunk)
    decoded_outputs_matched = gen_output(input_ids_matched, attention_mask_matched, 6)
    total_decoded_matched_outputs.extend(decoded_outputs_matched)

print("\nunmatched samples -- configure inputs; generate and decode outputs...")
total_decoded_unmatched_outputs = []
for chunk in chunked_unmatched_prompts:
    [tokenized_inputs_unmatched, input_ids_unmatched, attention_mask_unmatched] = tokenize_inputs(chunk)
    decoded_outputs_unmatched = gen_output(input_ids_unmatched, attention_mask_unmatched, 6)
    total_decoded_unmatched_outputs.extend(decoded_outputs_matched)

time_to_generate = datetime.now() - start_time
print("\ntime to generate outputs: ", time_to_generate)
print("\nmatched: ", total_decoded_matched_outputs[:10])
print("\nunmatched: ", total_decoded_unmatched_outputs[:10])


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

for i in range(len(total_decoded_matched_outputs)):
    total_decoded_matched_outputs[i] = verbalise(total_decoded_matched_outputs[i])

for i in range(len(total_decoded_unmatched_outputs)):
    total_decoded_unmatched_outputs[i] = verbalise(total_decoded_unmatched_outputs[i])

print("\n[verbalised] first ten matched: ", total_decoded_matched_outputs[:10])
print("\n[verbalised] first ten unmatched: ", total_decoded_unmatched_outputs[:10])

# 7. evaluation
matched_ground_truths = [sample[0] for sample in matched_samples]
unmatched_ground_truths = [sample[0] for sample in unmatched_samples]


print("matched accuracy: ", accuracy_score(matched_ground_truths, total_decoded_matched_outputs))
print("unmatched accuracy: ", accuracy_score(unmatched_ground_truths, total_decoded_unmatched_outputs))
