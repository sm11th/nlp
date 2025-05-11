# hey what is up you guys and welcome back to another video
import sys
import json

# import transformers, gpt2, etc
# from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer

# notebook_login()

# load access token from cmd line
# access_token = sys.argv[1]

# load model w/ appropriate decoding hyperparams (low max length, 0.7 temperature, etc)
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# load 2 datasets (matched + mismatched)
# figure out wtf matched vs mismatched settings are for (matched just means that some model had already seen the premise text's genre before)
# extract gold_label (gt), sentence 1, sentence 2 from datasets

matched_samples = []
with open("dev_matched_sampled-1.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)
        matched_samples.append((json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]))

# print("first 5 matched samples:\n")
# print(matched_samples[:5])

unmatched_samples = []
with open("dev_mismatched_sampled-1.jsonl", "r") as file:
    for line in file:
        json_line = json.loads(line)
        unmatched_samples.append((json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]))

# print("\n\nfirst 5 unmatched samples:\n")
# print(unmatched_samples[:5])

# function to generate inputs (prompt + sentences, put into gpt2, return gpt's output)
prompt = """
You are given two sentences: a premise and a hypothesis. Your task is to determine the relationship between them.
The relationship can be one of the following:
- Entailment: The hypothesis logically follows from the premise.
- Contradiction: The hypothesis logically contradicts the premise.
- Neutral: The hypothesis neither follows nor contradicts the premise.

Here is an example:
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Relationship:

"""

# prompt2 = """
# Premise: my friend Rob went to the grocery store where he bought some delicious apples.
# Hypothesis: my friend Rob has purchased some tasty apples.
# Result: entailment

# Premise: This morning I was told that I had been laid off from my job of twenty years.
# Hypothesis: I love my job too much to ever quit.
# Result: neutral

# Premise: The woman made her daughter cry.
# Hypothesis: The woman has never had a daughter.
# Result: contradiction

# Premise: {premise}
# Hypothesis: {hypothesis}
# Result: 
# """

def gen_input(line: tuple[str, str, str]) -> str:
    completed_prompt = prompt.format(premise=line[1], hypothesis=line[2])
    return completed_prompt

def gen_output(prompt: str) -> str:
    # which return_tensors do i actually want? this was taken from huggingface
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=4)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text

# create empty 'results' array
matched_results = []
unmatched_results = []

# for each row, do function to generate inputs, append output to 'results'
for row in matched_samples[:5]:
    prompt = gen_input(row)
    output = gen_output(prompt)
    matched_results.append(output)

# print out first 5 results
# TODO: fix this one
print("first 5 matched results:\n")
for result in matched_results[:5]:
    print(result[-20:])


for row in unmatched_samples[:5]:
    prompt = gen_input(row)
    output = gen_output(prompt)
    unmatched_results.append(output)

# print out first 5 results
# TODO: fix this one
print("\n\nfirst 5 unmatched results:\n")
for result in unmatched_results[:5]:
    print(result[-20:])


# save all results in csv


# create verbaliser (dictionary and such)

# for result in results: verbaliser.get(result)

# get accuracy, f1 score
