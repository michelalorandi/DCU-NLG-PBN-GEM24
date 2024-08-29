# general imports
import os
import json
import random
from typing import Optional, List
from dataclasses import dataclass, field

from tqdm import tqdm

# dataset related imports
from datasets import load_dataset

# model related imports
import torch
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)

from src.utils import post_process_function, preprocess_triples, extract_predicates, set_seed


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lan2text = {
    'en': 'English',
    'zh': 'Chinese',   
    'de': 'German',  
    'ru': 'Russian',   
    'es': 'Spanish',   
    'ko': 'Korean',   
    'hi': 'Hindi',   
    'sw': 'Swahili',   
    'ar': 'Arabic' 
}


@dataclass
class ScriptArguments:
    """
    Arguments for the execution of the script.
    """

    seed: Optional[int] = field(default=42)

    per_device_eval_batch_size: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=200)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-v0.1 ",
        metadata={
            "help": "The model that you want to load from the model folder."
        },
    )
    language: Optional[str] = field(
        default="en",
        metadata={
            "help": "Language to generate text in."
        },
    )
    prompt: Optional[str] = field(
        default="zero_shot",
        metadata={
            "help": "Prompt to generate text from."
        },
    )
    example_method: Optional[str] = field(
        default="fixed",
        metadata={
            "help": "Method to select examples from the dataset."
        },
    )
    triples_div: Optional[str] = field(
        default="; ",
        metadata={
            "help": "How to divide the triples in the input. E.g. ; or <br> etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="GEM/web_nlg",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_partition: Optional[str] = field(
        default="test",
        metadata={"help": "The partition of the dataset to use, e.g. test or validation."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )


def get_examples(input_triple_set: str | List,
                 method: str,
                 prompts_config: dict,
                 num_examples: int = 2) -> List[str]:
    """
    Get examples from the trainn set based on the specified method.

    Args:
        sample (str | List): input sample for which examples are to be selected.
        method (str): method to select examples. Can be 'fixed', 'length', 'predicates', 
                        'length_predicates', 'predicates_length'.
        prompts_config (dict): Configuration for the prompts.
        num_examples (int, optional): Number of examples to select. Defaults to 2.

    Raises:
        ValueError: If the specified method is not in the list of allowed methods.

    Returns:
        List[str]: List of examples selected based on the method.
    """
    if isinstance(input_triple_set, str):
        input_triple_set = input_triple_set.split('<br>')
    length = len(input_triple_set)
    predicates = extract_predicates(input_triple_set)

    if method == 'fixed':
        return prompts_config['few_shot']['fixed_examples']
    if method == 'length':
        # load saved train set by length and randomly select 2 examples with same length
        with open('data/data_length.json', 'r', encoding="utf-8") as file:
            data_length = json.load(file)
        return random.sample(data_length[str(length)], num_examples)
    if method == 'predicates':
        # load saved train set by predicates and randomly select 2 examples with same predicates
        with open('data/data_predicates.json', 'r', encoding="utf-8") as file:
            data_predicates = json.load(file)
        if len(predicates) == 0:
            predicates = data_predicates.keys()
        elif len(predicates) < num_examples:
            predicates = [predicates[0] for _ in range(num_examples)]
        selected_preds = random.sample(predicates, num_examples)
        selected_preds = [pred if pred in data_predicates else random.choice(data_predicates.keys())
                          for pred in selected_preds]
        return [random.choice(data_predicates[pred]) for pred in selected_preds]
    if method == 'length_predicates':
        # load saved train set by length and predicates and randomly select 2 examples
        # with same length and predicates
        with open('data/data_len_predicates.json', 'r', encoding="utf-8") as file:
            data_len_predicates = json.load(file)
        if len(predicates) == 0:
            predicates = data_len_predicates[str(length)].keys()
        elif len(predicates) < num_examples:
            predicates = [predicates[0] for _ in range(num_examples)]
        selected_preds = random.sample(predicates, num_examples)
        selected_preds = [pred if pred in data_len_predicates[str(length)]
                          else random.choice(list(data_len_predicates[str(length)].keys()))
                          for pred in selected_preds]
        return [random.choice(data_len_predicates[str(length)][pred]) for pred in selected_preds]
    if method == 'predicates_length':
        # load saved train set by predicates and load and randomly select 2 examples with
        # same predicates and length
        with open('data/data_predicates_len.json', 'r', encoding="utf-8") as file:
            data_len_predicates = json.load(file)

        if len(predicates) == 0:
            predicates = data_len_predicates[str(length)].keys()
        elif len(predicates) < num_examples:
            predicates = [predicates[0] for _ in range(num_examples)]

        selected_preds = random.sample(predicates, num_examples)
        selected_preds = [pred if pred in data_len_predicates
                          else random.choice(list(data_len_predicates.keys()))
                          for pred in selected_preds]
        examples = []
        for pred in selected_preds:
            if str(length) not in data_len_predicates[pred]:
                new_len = random.choice(list(data_len_predicates[pred].keys()))
                examples.append(random.choice(data_len_predicates[pred][new_len]))
            else:
                examples.append(random.choice(data_len_predicates[pred][str(length)]))
        return examples
    raise ValueError(f"Method {method} not recognized.")


def construct_prompt(input_triple_set: str | List,
                     prompts_config: dict,
                     triples_div: str,
                     lan: str,
                     prompt_type: str,
                     model_type: str,
                     example_method: str) -> str:
    """
    Construct the prompt based on the input triple set, the desired prompt, language and 
    the specified configuration.

    Args:
        input_triple_set (str | List): Set of triples to construct the prompt from.
        prompts_config (dict): Configuration for the prompts.
        triples_div (str): String to divide triples.
        lan (str): Language to generate text in.
        prompt_type (str): Type of prompt to generate. Can be 'zero_shot' or 'few_shot'.
        model_type (str): Type of model. Can be 'falcon' or 'mistral'.
        example_method (str): Method to select examples from the dataset.

    Returns:
        str: Constructed prompt.
    """
    start_instr = prompts_config['tokens'][model_type]['start_instr']
    end_instr = prompts_config['tokens'][model_type]['end_instr']
    start_ans = prompts_config['tokens'][model_type]['start_ans']

    if prompt_type == 'zero_shot':
        return prompts_config['zero_shot'].format(
            triples=preprocess_triples(input_triple_set, triples_div=triples_div),
            language=lan2text[lan],
            start_instr=start_instr,
            end_instr=end_instr,
            start_ans=start_ans)
    if prompt_type == 'few_shot':
        examples = get_examples(input_triple_set, example_method, prompts_config)

        examples = [prompts_config['few_shot']['example'].format(
            triples=preprocess_triples(example['input'], triples_div=triples_div),
            i=i+1,
            target=example['target'][lan]) for i, example in enumerate(examples)]
        examples = '\n##\n'.join(examples)

        return prompts_config['few_shot']['instruction'].format(
            triples=preprocess_triples(input_triple_set, triples_div=triples_div),
            language=lan2text[lan],
            start_instr=start_instr,
            end_instr=end_instr,
            start_ans=start_ans,
            examples=examples)
    return None


def preprocess_function(samples: dict,
                        config: dict,
                        triples_div:str,
                        lan: str,
                        prompt_type: str,
                        model_type: str,
                        example_method: str) -> dict:
    """
    Preprocesses the samples for the model to correctly populate the given template.

    Args:
        samples (dict): samples from the dataset with 'input' column.
        config (dict): Configuration for the prompts.
        triples_div (str): string to divide triples.
        lan (str): Language to generate text in.
        prompt_type (str): Type of prompt to generate. Can be 'zero_shot' or 'few_shot'.
        model_type (str): Type of model. Can be 'falcon' or 'mistral'.
        example_method (str): Method to select examples from the dataset.

    Returns:
        dict: Preprocessed samples as 'prompt' column.
    """
    pop_inputs = [construct_prompt(sample,
                                   config,
                                   triples_div,
                                   lan,
                                   prompt_type,
                                   model_type,
                                   example_method) for sample in samples['input']]
    return {'prompt': pop_inputs}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)

    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    random.seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                              trust_remote_code=True,
                                              padding=True,
                                              truncation=True,
                                              add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.use_4bit,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=script_args.use_nested_quant,
    )

    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True)

    model.eval()

    data_name = script_args.dataset_name.split('/')[-1].replace('.csv', '')

    if script_args.dataset_name.endswith('.csv'):
        eval_set = load_dataset("csv", data_files=script_args.dataset_name)['train']
    else:
        dataset = load_dataset(script_args.dataset_name, 'en')
        eval_set = dataset[script_args.dataset_partition]

    with open('prompts_config.json', 'r', encoding="utf-8") as config_file:
        prompts_config = json.load(config_file)

    eval_set = eval_set.map(preprocess_function,
                            batched=True,
                            fn_kwargs={'config': prompts_config,
                                       'triples_div': script_args.triples_div,
                                       'lan': script_args.language,
                                       'prompt_type': script_args.prompt,
                                       'model_type': 'falcon' if 'falcon' in script_args.model_name
                                       else 'mistral',
                                       'example_method': script_args.example_method})

    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token

    model_name = script_args.model_name.split('/')[1]
    eval_preds = []
    for i in tqdm(range(0, len(eval_set), script_args.per_device_eval_batch_size)):
        batch = eval_set[i:i+script_args.per_device_eval_batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch['prompt'], return_tensors="pt", padding=True).to("cuda:0")

            outputs = model.generate(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"],
                                     max_new_tokens=script_args.max_seq_length)

            decoded_output = tokenizer.batch_decode(outputs.detach().cpu().numpy())

        processed_outputs = post_process_function(decoded_output,
                                                  batch["prompt"],
                                                  sos_token=bos_token,
                                                  eos_token=eos_token)

        decoded_batch = [
            {
                "input": batch["prompt"][i], 
                "output": processed_outputs[i], 
                'references': batch['references'][i] if 'references' in eval_set.features else None,
                "raw_output": decoded_output[i]
            }
            for i in range(len(batch['prompt']))]
        eval_preds.extend(decoded_batch)

        predictions_file = f'predictions_{script_args.dataset_partition}_{data_name}_{model_name}_{script_args.prompt}_{script_args.example_method}_{script_args.language}.json'
        with open(os.path.join('outputs', predictions_file), 'w', encoding="utf-8") as out_file:
            json.dump(eval_preds, out_file)
