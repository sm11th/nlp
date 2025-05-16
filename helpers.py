import json

# this one is kind of dataset-specific right now
def extract_nli_data(json_name: str) -> list:
    print("\nextracting data...")
    samples = []
    with open(json_name, "r") as file:
        for line in file:
            json_line = json.loads(line)
            samples.append([json_line["gold_label"], json_line["sentence1"], json_line["sentence2"]])
    return samples


# format the prompt with the correct 'premise' and 'hypothesis'
def gen_input(prompt: str, premise: str, hypothesis: str) -> str:
    formatted_prompt = prompt + " Premise: " + premise + " Hypothesis: " + hypothesis + " Relationship: "
    return formatted_prompt


# set gpt2 model w/ appropriate decoding hyperparams
def gen_gpt2_output(prompt: str, model, tokenizer, max_tokens: int) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
    input_ids,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=max_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


def gen_t5_output(prompts, tokenizer, model, max_length) -> list:
    [input_ids, attention_mask] = tokenize_inputs(prompts, tokenizer)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded_output


def tokenize_inputs(prompts, tokenizer):
    tokenized_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"].to("mps")  # move tensors to MPS
    attention_mask = tokenized_inputs["attention_mask"].to("mps")
    return [input_ids, attention_mask]


# create verbaliser (converting output and such)
def verbalise_inf(output: str) -> str:
    output = output.lower()
    entailment_strings = ["entail", "entailment", "true", "correct", "same", "does hold"]
    contradiction_strings = ["contradict", "false", "wrong", "different", "does not hold"]
    neutral_strings = ["neutral", "ambiguous", "maybe"]
    if(any(word in output for word in entailment_strings)):
        return "entailment"
    elif(any(word in output for word in contradiction_strings)):
        return "contradiction"
    elif(any(word in output for word in neutral_strings)):
        return "neutral"
    else:
        return "no meaningful result"
    
def verbalise_halluc(output: str) -> int:
    output = output.lower()
    nonfactual_strings = ["0", "non-factual", "nonfactual", "non factual", "inaccurate", "contradict", "false", "wrong", "different"]
    if(any(word in output for word in nonfactual_strings)):
        return 0
    else:
        return 1

def verbalise_list(list: list[str], task: str) -> list:
    verbalised_list = []
    if task == "entailment":
        for i in range(len(list)):
            output = list[i]
            verbalised_output = verbalise_inf(output)
            verbalised_list.append(verbalised_output)
    elif task == "fact-checking":
        for i in range(len(list)):
            output = list[i]
            verbalised_output = verbalise_halluc(output)
            verbalised_list.append(verbalised_output)
    else:
        raise NotImplementedError("{task} has not yet been implemented in verbalise_list")
    return verbalised_list

def get_pseudo_log_likelihood():
    print("hello")