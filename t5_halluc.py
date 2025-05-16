import torch
import os
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helpers import gen_input, verbalise_list, extract_nli_data, gen_t5_output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # suppress transformer warnings
warnings.filterwarnings('ignore')  # suppress all other warnings

# same type situation
# load the data
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split='evaluation')

# model setup
mps_device = torch.device("mps")  # mps = runs on apple silicon gpus instead of just cpus

print("loading model...")
model = T5ForConditionalGeneration.from_pretrained("t5-small", device_map=str("auto")).to(mps_device)

# tokenizer setup
print("\nloading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# make prefix:
prefix = """
Factuality classification (1 = factual, 0 = non-factual):
Is the hypothesis factually supported by the premise? Answer with 1 (yes) or 0 (no):
"""

# shard (chunk) the data
# divide list into 58 shards (should be around 4 elements per shard) (my laptop cannot handle any more, RIP)
num_shards = 50

total_outputs = []
total_ground_truths = []

for i in range(num_shards):
    current_shard = dataset.shard(num_shards=num_shards, index=i)
    total_prompts = []

    for line in current_shard:
        wiki_text = line["wiki_bio_text"]
        gpt_sentences = line["gpt3_sentences"]
        line_ground_truths = line["annotation"]
        total_ground_truths.extend(line_ground_truths)
        sentence_outputs = []

        for sentence in gpt_sentences:
            formatted_prompt = gen_input(prompt=prefix, premise=wiki_text, hypothesis=sentence)
            total_prompts.append(formatted_prompt)
    
    outputs = gen_t5_output(total_prompts, tokenizer, model, max_length=6)
    total_outputs.extend(outputs)
    

verbalised_outputs = verbalise_list(total_outputs, "fact-checking")
verbalised_ground_truths = verbalise_list(total_ground_truths, "fact-checking")

print("accuracy: ", accuracy_score(verbalised_ground_truths, verbalised_outputs))
print("precision: ", precision_score(verbalised_ground_truths, verbalised_outputs))
print("recall: ", recall_score(verbalised_ground_truths, verbalised_outputs))
print("f1 score: ", f1_score(verbalised_ground_truths, verbalised_outputs))

