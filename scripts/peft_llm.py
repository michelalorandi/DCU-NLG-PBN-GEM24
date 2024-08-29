# General imports
import os
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

import wandb

# Hugging Face imports
# models
import torch
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    PrefixTuningConfig,
)
from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)

# datasets
from datasets import load_dataset

# training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.utils import set_seed


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["WANDB_PROJECT"] = "GEM_PEFT_project"
os.environ["WANDB_LOG_MODEL"] = (
    "end"  # Save the model on Wandb as artifact at the end of training
)

WANDB_PROJECT = "GEM_PEFT_project"


@dataclass
class ScriptArguments:
    """
    Arguments for the execution of the script.
    """

    seed: Optional[int] = field(default=42)

    per_device_train_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-v0.1 ",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
            "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    template: Optional[str] = field(
        default="[INST]Write the following triples as fluent English text. "
        'Triples: """ {triples} """ [/INST] {output}',
        metadata={
            "help": "Representation of the input triples. E.g. Triples: or Input: etc."
        },
    )
    text_column: Optional[str] = field(
        default="Triples: ",
        metadata={
            "help": "Representation of the input triples. E.g. Triples: or Input: etc."
        },
    )
    triples_div: Optional[str] = field(
        default="; ",
        metadata={
            "help": "How to divide the triples in the input. E.g. ; or <br> etc."
        },
    )
    label_column: Optional[str] = field(
        default=". Text:",
        metadata={
            "help": "Representation of the output text. E.g. . Text: or . Output: etc."
        },
    )
    peft_type: Optional[str] = field(
        default="prompt",
        metadata={"help": "the type of PEFT to use. E.g. prompt, lora etc."},
    )
    dataset_name: Optional[str] = field(
        default="GEM/web_nlg",
        metadata={"help": "The preference dataset to use."},
    )
    padding: Optional[str] = field(
        default="right",
        metadata={"help": "The padding side for the input."},
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
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, "
            "and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds "
            "up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )


def create_and_prepare_model(
    args: ScriptArguments,
) -> Tuple[AutoModelForCausalLM, Optional[PeftConfig], AutoTokenizer]:
    """
    Create and prepare the model for fine-tuning using quantization and the specified PEFT
    technique.

    Args:
        args (ScriptArguments): Arguments to load the model with correct hyperparameters.

    Returns:
        Tuple[AutoModelForCausalLM, Optional[PeftConfig], AutoTokenizer]: model, configuration
                of PEFT and tokenizer
    """
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    peft_config = None
    if args.peft_type == "lora":
        if "Mistral" in args.model_name:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
        else:
            target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    elif args.peft_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="Write the following triples as fluent text:",
            tokenizer_name_or_path=args.model_name,
        )
        model = get_peft_model(model, peft_config)
    elif args.peft_type == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, prefix_projection=False
        )
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side=args.padding,
        add_eos_token=True,
    )
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]"}
    )  # add a new pad token to the tokenizer

    return model, peft_config, tokenizer


def preprocess_triples(row: dict, triples_div: str = "; ") -> dict:
    """
    Preprocess the triples to convert them into a single string in the format
    'subject predicate object' divided by the specified triples divder.

    Args:
        row (dict): The row containing the triples.
        triples_div (str, optional): The divider for the triples. Defaults to "; ".

    Returns:
        dict: The processed triples.
    """
    return {"triples": triples_div.join(row["input"]).replace(" | ", " ")}


def formatting_func(samples: dict) -> List[str]:
    """
    Format the samples using the template provided in the script arguments to feed
    them to the model.

    Args:
        samples (dict): The samples from the dataset.

    Returns:
        List[str]: The formatted prompts.
    """
    prompts = [
        script_args.template.format(
            triples=samples["triples"][i], output=samples["target"][i]
        )
        for i in range(len(samples["target"]))
    ]
    return prompts


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)

    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    mod_name = script_args.model_name.split("/")[-1]
    experiment_name = f"{mod_name}_peft_{script_args.peft_type}_paddingLeft_instrSent_maxsteps{script_args.max_steps}_seed{script_args.seed}"

    # set the wandb experiment in the correct project
    print(f"Setting up W&B experiment {experiment_name}")
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=experiment_name,
        tags=[script_args.model_name, script_args.peft_type, script_args.dataset_name],
    )

    wandb.config.update(
        {
            "text_column": script_args.text_column,
            "label_column": script_args.label_column,
            "dataset_name": script_args.dataset_name,
            "template": script_args.template,
        }
    )

    # Load the model and tokenizer
    print("Loading model and tokenizer")
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False

    # Load the dataset
    print("Loading dataset")
    dataset = load_dataset(script_args.dataset_name, "en").map(preprocess_triples)

    response_template = script_args.label_column
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    experiment_folder = os.path.join("models", experiment_name)
    with open(
        os.path.join(experiment_folder, "exp_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(script_args.__dict__, f)

    # Set up the training arguments
    print("Setting up training arguments")
    training_arguments = TrainingArguments(
        output_dir=experiment_name,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,  # how often to log to W&B
        learning_rate=script_args.learning_rate,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        group_by_length=script_args.group_by_length,
        lr_scheduler_type=script_args.lr_scheduler_type,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_total_limit=2,
        report_to="wandb",  # enable logging to W&B
        run_name=experiment_name,  # name of the W&B run (optional)
    )

    # Set up the trainer
    print("Setting up the trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=script_args.packing,
        formatting_func=formatting_func,
    )

    # Set up the model to use the correct precision
    for name, module in trainer.model.named_modules():
        if script_args.peft_type == "lora":
            if isinstance(module, LoraLayer):
                if script_args.bf16:
                    module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if script_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # Train the model
    print("Training the model")
    trainer.train()

    wandb.finish()
