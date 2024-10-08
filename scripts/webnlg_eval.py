__author__='thiagocastroferreira,liamcripwell'

"""
Author: Organizers of the 3rd WebNLG Challenge
Date: 21/02/2023
Description:
    This script aims to evaluate the output of data-to-text NLG models by computing 
    popular automatic metrics such as BLEU, METEOR, chrF++, TER and BERT-Score.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-lng LANGUAGE] [-nr NUM_REFS]
               [-m METRICS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

        optional arguments:
          -h, --help            show this help message and exit
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H HYPOTHESIS, --hypothesis HYPOTHESIS
                                hypothesis translation
          -lng LANGUAGE, --language LANGUAGE
                                evaluated language
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references
          -m METRICS, --metrics METRICS
                                evaluation metrics to be computed
          -nc NCORDER, --ncorder NCORDER
                                chrF metric: character n-gram order (default=6)
          -nw NWORDER, --nworder NWORDER
                                chrF metric: word n-gram order (default=2)
          -b BETA, --beta BETA  chrF metric: beta parameter (default=2)

    EXAMPLE:
        ENGLISH: 
            python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt
        RUSSIAN:
            python3 eval.py -R data/ru/reference -H data/ru/hypothesis -lng ru -nr 1 -m bleu,meteor,chrf++,ter,bert
"""

import argparse
import codecs
import copy
import os
import pyter
import logging
import nltk
import json
import subprocess
import re

from bert_score import score
from chrF import computeChrF
from razdel import tokenize
from tabulate import tabulate
from sacrebleu.metrics import BLEU

METEOR_PATH = 'metrics/meteor-1.5/meteor-1.5.jar'

BERT_LANGS = ["br", "cy", "en", "ga", "ru"]
METEOR_LANGS = ["en", "ru"]


def parse(refs_path, hyps_path, num_refs, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            #texts = f.read().split('\n')
            texts = [x.strip() for x in f.readlines()]
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)

    # references tokenized
    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if lng == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        #hypothesis = f.read().split('\n')
        hypothesis = [x.strip() for x in f.readlines()]

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    if lng == 'ru':
        hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
    else:
        hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]

    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok


def sacrebleu_score(references, hypothesis, num_refs):
    refs = []
    for i in range(num_refs):
        # allow for variable number of references per example
        refs.append([ref[i] if len(ref) > i and ref[i].strip() != "" else None for ref in references])

    bleu = BLEU()
    return bleu.corpus_score(hypothesis, refs).score


def meteor_score(references, hypothesis, num_refs, lng='en'):
    logging.info('STARTING TO COMPUTE METEOR...')
    print('STARTING TO COMPUTE METEOR...')
    hyps_tmp, refs_tmp = 'hypothesis_meteor', 'reference_meteor'

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis))

    linear_references = []
    for refs in references:
        for i in range(num_refs):
          if len(refs) > i:
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    try:
        command = 'java -Xmx2G -jar {0} '.format(METEOR_PATH)

        # handle unsupported langs
        if lng not in METEOR_LANGS:
            print(f"Language {lng} not officially supported by METEOR. Reverting to language-independent version.")
            lng = "other"

        if lng == "other":
            command += '{0} {1} -l {2} -r {3}'.format(hyps_tmp, refs_tmp, lng, num_refs)
        else:
            command += '{0} {1} -l {2} -norm -r {3}'.format(hyps_tmp, refs_tmp, lng, num_refs)
        print(f"METEOR command executed: {command}")
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b'\n')[-2].split()[-1]
    except:
        logging.error('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        meteor = -1

    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE METEOR...')
    print('FINISHING TO COMPUTE METEOR...')
    return float(meteor)


def chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta):
    logging.info('STARTING TO COMPUTE CHRF++...')
    print('STARTING TO COMPUTE CHRF++...')
    hyps_tmp, refs_tmp = 'hypothesis_chrF', 'reference_chrF'

    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            hypothesis_.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, nworder, ncorder, beta, None)
    except:
        logging.error('ERROR ON COMPUTING CHRF++.')
        print('ERROR ON COMPUTING CHRF++.')
        totalF, averageTotalF, totalPrec, totalRec = -1, -1, -1, -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE CHRF++...')
    print('FINISHING TO COMPUTE CHRF++...')
    return totalF, averageTotalF, totalPrec, totalRec


def ter_score(references, hypothesis, num_refs):
    logging.info('STARTING TO COMPUTE TER...')
    print('STARTING TO COMPUTE TER...')
    ter_scores = []
    for hyp, refs in zip(hypothesis, references):
        candidates = []
        for ref in refs[:num_refs]:
            if len(ref) == 0:
                ter_score = 1
            else:
                try:
                    ter_score = pyter.ter(hyp.split(), ref.split())
                except:
                    ter_score = 1
            candidates.append(ter_score)

        ter_scores.append(min(candidates))

    logging.info('FINISHING TO COMPUTE TER...')
    print('FINISHING TO COMPUTE TER...')
    return sum(ter_scores) / len(ter_scores)


def bert_score_(references, hypothesis, lng='en'):
    logging.info('STARTING TO COMPUTE BERT SCORE...')
    print('STARTING TO COMPUTE BERT SCORE...')
    for i, refs in enumerate(references):
        references[i] = [ref for ref in refs if ref.strip() != '']

    try:
        if lng not in BERT_LANGS:
            print(f"Language {lng} not officially supported by BERT Score metric.")
        P, R, F1 = score(hypothesis, references, lang=lng)
        logging.info('FINISHING TO COMPUTE BERT SCORE...')
    #     print('FINISHING TO COMPUTE BERT SCORE...')
        P, R, F1 = list(P), list(R), list(F1)
        F1 = float(sum(F1) / len(F1))
        P = float(sum(P) / len(P))
        R = float(sum(R) / len(R))
    except:
        print("BERTScore calculation failed... setting to default value of 0.")
        P, R, F1 = 0, 0, 0
    return P, R, F1


def run(refs_path, hyps_path, num_refs, lng='en', metrics='bleu,meteor,chrf++,ter,bert,bleurt',ncorder=6, nworder=2, beta=2):
    metrics = metrics.lower().split(',')
    references, references_tok, hypothesis, hypothesis_tok = parse(refs_path, hyps_path, num_refs, lng)

    result = {}
    
    logging.info('STARTING EVALUATION...')
    if 'bleu' in metrics:
        bleu = sacrebleu_score(references, hypothesis, num_refs)
        result["bleu"] = bleu
    if 'meteor' in metrics:
        meteor = meteor_score(references_tok, hypothesis_tok, num_refs, lng=lng)
        result['meteor'] = meteor
    if 'chrf++' in metrics:
        chrf, _, _, _ = chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta)
        result['chrf++'] = chrf
    if 'ter' in metrics:
        ter = ter_score(references_tok, hypothesis_tok, num_refs)
        result['ter'] = ter
    if 'bert' in metrics:
        P, R, F1 = bert_score_(references, hypothesis, lng=lng)
        result['bert_precision'] = P
        result['bert_recall'] = R
        result['bert_f1'] = F1
    if 'bleurt' in metrics and lng == 'en':
        s = bleurt(references, hypothesis, num_refs)
        result['bleurt'] = s
    logging.info('FINISHING EVALUATION...')
    
    return result


def main():
    FORMAT = '%(levelname)s: %(asctime)-15s - %(message)s'
    logging.basicConfig(filename='eval.log', level=logging.INFO, format=FORMAT)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-ref", "--reference", help="reference translation", required=True)
    argParser.add_argument("-hyp", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-lng", "--language", help="evaluated language", default='en')
    argParser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=4)
    argParser.add_argument("-m", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,ter,chrf++,bert,bleurt')
    argParser.add_argument("-nc", "--ncorder", help="chrF metric: character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="chrF metric: word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="chrF metric: beta parameter (default=2)", type=float, default=2.0)

    args = argParser.parse_args()

    logging.info('READING INPUTS...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    lng = args.language
    num_refs = args.num_refs
    metrics = args.metrics#.lower().split(',')

    nworder = args.nworder
    ncorder = args.ncorder
    beta = args.beta
    logging.info('FINISHING TO READ INPUTS...')

    result = run(refs_path=refs_path, hyps_path=hyps_path, num_refs=num_refs, lng=lng, metrics=metrics, ncorder=ncorder, nworder=nworder, beta=beta)
    
    with open(hyps_path.replace('output_validation', 'results').replace('.txt', '.json'), 'w') as res_file:
      json.dump(result, res_file)

    metrics = metrics.lower().split(',')
    headers, values = [], []
    if 'bleu' in metrics:
        headers.append("BLEU")
        values.append(result['bleu'])
    if 'meteor' in metrics:
        headers.append('METEOR')
        values.append(round(result['meteor'], 2))
    if 'chrf++' in metrics:
        headers.append('chrF++')
        values.append(round(result['chrf++'], 2))
    if 'ter' in metrics:
        headers.append('TER')
        values.append(round(result['ter'], 2))
    if 'bert' in metrics:
        headers.append('BERT-SCORE P')
        values.append(round(result['bert_precision'], 2))
        headers.append('BERT-SCORE R')
        values.append(round(result['bert_recall'], 2))
        headers.append('BERT-SCORE F1')
        values.append(round(result['bert_f1'], 2))
    if 'bleurt' in metrics and lng == 'en':
        headers.append('BLEURT')
        values.append(round(result['bleurt'], 2))

    logging.info('PRINTING RESULTS...')
    print(tabulate([values], headers=headers))

if __name__ == '__main__':
    main()
