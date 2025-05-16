from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helpers import gen_input, gen_gpt2_output, verbalise_list

# load dataset
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split='evaluation')

# load model 
print("load model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print("\nload tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# make prompt
prompt = """
I am going to show you two texts: a premise and a hypothesis. I'll explain the relationship between them.
The relationship can be one of the following:
- Factual: the hypothesis is factual given the premise
- Nonfactual: the hypothesis is not factual given the premise

Here are the sentences with their relationship:
"""

total_outputs = []
ground_truths = []

for line in dataset:
    wiki_text = line["wiki_bio_text"]
    gpt_sentences = line["gpt3_sentences"]
    line_ground_truths = line["annotation"]

    sentence_outputs = []
    for sentence in gpt_sentences:
        formatted_prompt = gen_input(prompt=prompt, premise=wiki_text,  hypothesis=sentence)  # format prompt
        output = gen_gpt2_output(prompt=formatted_prompt, model=model, tokenizer=tokenizer, max_tokens=10)  # get results
        cleaned_output = output.split("Relationship: ", 1)[1]  # split into prompt ([0]) and output ([1]), get just output
        sentence_outputs.append(cleaned_output)

    verbalised_sentence_outputs = verbalise_list(sentence_outputs, "fact-checking")
    verbalised_line_ground_truths = verbalise_list(line_ground_truths, "fact-checking")

    ground_truths.extend(verbalised_line_ground_truths)
    total_outputs.extend(verbalised_sentence_outputs)


print(total_outputs[0])

# evaluation
print("accuracy: ", accuracy_score(ground_truths, total_outputs))
print("precision: ", precision_score(ground_truths, total_outputs))
print("recall: ", recall_score(ground_truths, total_outputs))
print("f1 score: ", f1_score(ground_truths, total_outputs))
