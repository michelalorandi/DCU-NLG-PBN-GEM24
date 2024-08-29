from typing import List

import torch


def set_seed(seed: int):
    """
    Set the seed for reproducibility in torch.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_predicates(triples: List[str]) -> List[str]:
    """
    Extract predicates from the triples.

    Args:
        triples (List[str]): The list of triples.

    Returns:
        List[str]: The list of predicates.
    """
    predicates = []
    for triple in triples:
        predicates.append(triple.split(' | ')[1])
    return predicates


def preprocess_triples(triples: str | List[str], triples_div: str = "; ") -> str:
    """
    Preprocess the triples.

    Args:
        triples (str | List[str]): Triples to preprocess.
        triples_div (str, optional): The divider for the triples. Defaults to "; ".

    Returns:
        str: The preprocessed triples.
    """
    if isinstance(triples, str):
        triples = triples.split('<br>')
    return triples_div.join(triples).replace(" | ", " ")


def post_process_function(outputs: List[str],
                          prompts: List[str],
                          final_tag: str = None,
                          sos_token: str = '<s>',
                          eos_token: str = '</s>') -> List[str]:
    """
    Post-process the outputs keeping the text until the specified final tag, removes sos and 
    eos tokens, removes [ and ].

    Args:
        outputs (List[str]): Outputs from the model to be processed.
        prompts (List[str]): Prompts used in input.
        final_tag (str, optional): Final tag to keep the text until. Defaults to None.
        sos_token (str, optional): _Start of sentence token_. Defaults to '<s>'.
        eos_token (str, optional): End of sentence token. Defaults to '</s>'.

    Returns:
        List[str]: The processed outputs.
    """
    final_outputs = []
    for i, prompt in enumerate(prompts):
        output = outputs[i]
        output = output.replace(prompt, "")

        if final_tag is not None:
            output = output.split(final_tag)[0]

        if sos_token is not None:
            output = output.replace(sos_token, '')
        if eos_token is not None:
            output = output.replace(eos_token, '')

        output = output.split('Triples')[0].split('(Note:')[0]

        output = output.replace(']','').replace('[', '').replace('\n', ' ').strip()
        final_outputs.append(output)
    return final_outputs
