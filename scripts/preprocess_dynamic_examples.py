
import os
import json
from typing import List

import torch

from tqdm import tqdm

from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import extract_predicates


BATCH_SIZE = 512

data_folder = os.path.join('data', 'dynamic_examples')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lang2code = {
    'zh': 'zho_Hans',   # Chinese
    'de': 'deu_Latn',   # German
    'ru': 'rus_Cyrl',   # Russian
    'es': 'spa_Latn',   # Spanish
    'ko': 'kor_Hang',   # Korean
    'hi': 'hin_Deva',   # Hindi
    'sw': 'swh_Latn',   # Swahili
    'ar': 'arb_Arab'    # Arabic
}


def translate_batch(text: List[str],
                    lang: str,
                    model: AutoModelForSeq2SeqLM,
                    tokenizer: AutoTokenizer) -> List[str]:
    """
    Translate a batch of text to the target language using the given model.

    Args:
        text (List[str]): The list of texts to translate.
        lang (str): The target language.
        model (AutoModelForSeq2SeqLM): The model to use for translation.
        tokenizer (AutoTokenizer): The tokenizer to use for translation.

    Returns:
        List[str]: The list of translated texts.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[lang2code[lang]]
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
    model.to(device)

    dataset = load_dataset("GEM/web_nlg", 'en')['train']

    translated_data = {
            'input': dataset['input'],
            'target_en': dataset['target']
    }
    for key in lang2code:
        translated_data[f'target_{key}']=[]
        for index in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[index:index+BATCH_SIZE]
            translations = translate_batch(batch['target'], key, model, tokenizer)
            translated_data[f'target_{key}'] += translations

    dataset = Dataset.from_dict(translated_data)

    data_length = {}
    data_predicates = {}
    data_len_predicates = {}
    data_predicates_len = {}

    for sample in dataset:
        triples_len = len(sample['input'])
        predicates = extract_predicates(sample['input'])

        s = {'input': sample['input'], 'target': {'en': sample['target_en']}}
        for key in lang2code:
            s['target'][key] = sample[f'target_{key}']

        if triples_len not in data_length:
            data_length[triples_len] = []
        data_length[triples_len].append(s)

        if triples_len not in data_len_predicates:
            data_len_predicates[triples_len] = {}

        for predicate in predicates:
            if predicate not in data_predicates:
                data_predicates[predicate] = []

            if predicate not in data_predicates_len:
                data_predicates_len[predicate] = {}
            if triples_len not in data_predicates_len[predicate]:
                data_predicates_len[predicate][triples_len] = []

            if predicate not in data_len_predicates[triples_len]:
                data_len_predicates[triples_len][predicate] = []

            data_predicates[predicate].append(s)
            data_predicates_len[predicate][triples_len].append(s)
            data_len_predicates[triples_len][predicate].append(s)

    with open(os.path.join(data_folder, 'data_length.json'), 'w', encoding="utf-8") as f:
        json.dump(data_length, f)

    with open(os.path.join(data_folder, 'data_predicates.json'), 'w', encoding="utf-8") as f:
        json.dump(data_predicates, f)

    with open(os.path.join(data_folder, 'data_predicates_len.json'), 'w', encoding="utf-8") as f:
        json.dump(data_predicates_len, f)

    with open(os.path.join(data_folder, 'data_len_predicates.json'), 'w', encoding="utf-8") as f:
        json.dump(data_len_predicates, f)
