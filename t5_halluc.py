import torch
import os
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helpers import gen_input, verbalise_list, extract_nli_data, get_t5_output
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # suppress transformer warnings
warnings.filterwarnings('ignore')  # suppress all other warnings

# same type situation
# load the data
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split='evaluation').select(range(10))


# load the model

# load the tokenizer

# make prompt w/ gen_input + prefix

# chunk the data

# total_outputs = []
# total_ground_truths = []
# for each chunk in data:
    # sentences_outputs = []
    # for each sentence in chunk:
        # get_t5_output
        # append output to sentences_outputs
    
    # verbalise sentences_outputs
    # verbalise ground truths in chunk

    # total_outputs.extend(verbalised sentence outputs)
    # total_ground_truths.extend(verbalised ground truths)

# print("accuracy: ", accuracy_score(ground_truths, total_outputs))
# print("precision: ", precision_score(ground_truths, total_outputs))
# print("recall: ", recall_score(ground_truths, total_outputs))
# print("f1 score: ", f1_score(ground_truths, total_outputs))

