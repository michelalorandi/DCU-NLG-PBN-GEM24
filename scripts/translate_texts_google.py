import os
import json
import argparse
from os import environ

from tqdm import tqdm

from google.cloud import translate


PROJECT_ID = environ.get("PROJECT_ID", "")
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"


def translate_text(text: str, target_language_code: str) -> translate.Translation:
    """
    Translates the text to the target language.

    Args:
        text (str): Text to translate.
        target_language_code (str): The target language code.

    Returns:
        translate.Translation: The translation response.
    """
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        source_language_code="en",
        target_language_code=target_language_code,
        mime_type="text/plain"
    )

    return response.translations[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', metavar='path', required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--out_filepath', metavar='path', required=True)
    args = parser.parse_args()

    with open(args.filepath, 'r', encoding="utf-8") as f:
        outputs = json.load(f)

    translated_data = []
    if os.path.exists(args.out_filepath):
        with open(args.out_filepath, 'r', encoding="utf-8") as f:
            translated_data = json.load(f)
    start = max(0, len(translated_data))

    for index in tqdm(range(start, len(outputs)),
                      desc=f"Translating {args.filepath} to {args.lang}"):
        out = outputs[index]

        translation = translate_text(out['output'], args.lang)
        source_language = translation.detected_language_code
        translated_text = translation.translated_text

        translated_data.append({
            'input': out['input'],
            'original_output': out['output'],
            'output': translated_text,
            'references': out['references'],
            'raw_output': out['raw_output']
        })

        with open(args.out_filepath, 'w', encoding="utf-8") as f:
            json.dump(translated_data, f)
