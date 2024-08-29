# general imports
import os
import json
from typing import Optional
from dataclasses import dataclass, field

from tqdm import tqdm

# dataset related imports
from datasets import load_dataset

# model related imports
import torch
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

from src.utils import post_process_function, preprocess_triples


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Set the seed
def set_seed(seed: int):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class ScriptArguments:
    """
    Arguments for the execution of the script.
    """

    seed: Optional[int] = field(default=42)

    per_device_eval_batch_size: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=200)
    model_folder: Optional[str] = field(
        default=".",
        metadata={"help": "Folder containing all models."},
    )
    model_name: Optional[str] = field(
        default="Mistral-7B-Instruct-v0.2_peft_lora_paddingRight_instrSent_maxsteps10000_seed6787",
        metadata={"help": "The model that you want to load from the model folder."},
    )
    model_checkpoint: Optional[str] = field(
        default="checkpoint_4000",
        metadata={
            "help": "The checkpoint that you want to load from the model folder."
        },
    )
    dataset_name: Optional[str] = field(
        default="GEM/web_nlg",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_partition: Optional[str] = field(
        default="test",
        metadata={
            "help": "The partition of the dataset to use, e.g. test or validation."
        },
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


def preprocess_function(samples: dict, input_template: str, triples_div: str) -> dict:
    """
    Preprocesses the samples for the model to correctly populate the given template.

    Args:
        samples (dict): samples from the dataset with 'input' column.
        input_template (str): string template for the model input.
        triples_div (str): string to divide triples.

    Returns:
        dict: _description_
    """
    pop_inputs = [
        input_template.format(triples=preprocess_triples(x, triples_div=triples_div))
        for x in samples["input"]
    ]
    return {"prompt": pop_inputs}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    model_folder = os.path.join(script_args.model_folder, script_args.model_name)
    peft_model_id = os.path.join(model_folder, script_args.model_checkpoint)
    exp_config_filepath = os.path.join(model_folder, "exp_config.json")

    with open(exp_config_filepath, "r", encoding="utf-8") as f:
        exp_config = json.load(f)

    config = PeftConfig.from_pretrained(peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        peft_model_id,
        trust_remote_code=True,
        padding=True,
        truncation=True,
        add_eos_token=False,
    )
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
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    model.to(DEVICE)
    model.eval()

    data_name = script_args.dataset_name.split("/")[-1].replace(".csv", "")

    if script_args.dataset_name.endswith(".csv"):
        dataset = load_dataset("csv", data_files=script_args.dataset_name)["train"]
    else:
        dataset = load_dataset(script_args.dataset_name, "en")

    template = exp_config["template"].split("{output}")[0]
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"template": template, "triples_div": exp_config["triples_div"]},
    )
    if script_args.dataset_name.endswith(".csv"):
        eval_set = dataset
    else:
        eval_set = dataset[script_args.dataset_partition]

    final_tag = exp_config["template"].split("{output}")[-1].strip()

    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token

    eval_preds = []
    for i in tqdm(range(0, len(eval_set), script_args.per_device_eval_batch_size)):
        batch = eval_set[i : i + script_args.per_device_eval_batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True).to(DEVICE)

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=script_args.max_seq_length,
            )
            outputs = outputs.detach().cpu().numpy()
            decoded_output = tokenizer.batch_decode(outputs)

        processed_outputs = post_process_function(
            decoded_output,
            batch["prompt"],
            final_tag=final_tag,
            sos_token=bos_token,
            eos_token=eos_token,
        )

        decoded_batch = [
            {
                "input": batch["prompt"][i],
                "output": processed_outputs[i],
                "references": (
                    batch["references"][i]
                    if "references" in eval_set.features
                    else None
                ),
                "raw_output": decoded_output[i],
            }
            for i in range(len(batch["prompt"]))
        ]
        eval_preds.extend(decoded_batch)

        prediction_file = f"predictions_{script_args.dataset_partition}_{data_name}_{script_args.model_name}.json"
        with open(os.path.join("outputs", prediction_file), "w", encoding="utf-8") as f:
            json.dump(eval_preds, f)
