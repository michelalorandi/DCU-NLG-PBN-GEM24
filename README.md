# GEM 2024 Data-to-Text Task: DCU-NLG-PBN Submission

This repository contains the code, models, and generated outputs for our submission to the GEM 2024 Data-to-Text Task. Our work focuses on data-to-text generation using Large Language Models (LLMs) using prompting and fine-tuning.

## Overview

In this project, we explored two primary approaches for data-to-text generation:

1. **LLM Prompting**: Utilizing out-of-the-box LLMs with few-shot in-context learning, enhanced by dynamic example selection based on the input structure.
2. **LLM Fine-Tuning with LoRA**: Fine-tuning LLMs using Parameter Efficient Fine-Tuning (PEFT) techniques to improve performance while reducing computational costs.

## Project Structure

- `scripts/`: Contains all the scripts used for:
  - Prompting LLMs
  - Fine-tuning LLMs using LoRA
  - Post-processing and translation of generated outputs
  - Automatic evaluation of results

- `models/`: Fine-tuned models used in this project. Download them here.

- `data/`: Placeholder for datasets used in the experiments. Note that actual datasets might need to be downloaded or generated separately.

- `outputs/`: Generated text outputs from the LLMs in various languages, post-processed and translated.

## Setup

### Environments creation

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- Required Python packages for evaluation (see `eval_requirements.txt`)
- Access to the Google Translate API for multilingual translation

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/michelalorandi/DCU-NLG-PBN-GEM24
   cd DCU-NLG-PBN-GEM24
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   or 

   ```bash
   conda create --name peft_llm_env --file requirements.txt
   ```

3. Install the required packages for evaluation:

   ```bash
   pip install -r eval_requirements.txt
   ```

   or 

   ```bash
   conda create --name eval_env --file eval_requirements.txt
   ```

4. Set up access to the Google Translate API (if multilingual translation is needed).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions or issues, please contact [michela.lorandi@adaptcentre.ie](mailto:michela.lorandi@adaptcentre.ie).



