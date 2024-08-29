import os
import xml.etree.ElementTree as ET

import pandas as pd


data_dir = os.path.join('data', 'GEM-v2-D2T-SharedTask')


# Load data file and put it in json format
def get_data(filepath: str) -> list[dict[str, str]]:
    """
    Load data file and put it in json format.

    Args:
        filepath (str): The path to the file.

    Returns:
        list[dict[str, str]]: The data in json format.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    data = []
    # Extract all entries
    for entry in root.findall('./entries/entry'):
        triples = []
        # Extract the triples associated to the current entry
        for triple in entry.find('modifiedtripleset').findall('mtriple'):
            triples.append(triple.text)

        data.append({
            'input': '<br>'.join(triples),  # List of triples (string)
        })
    return data


if __name__ == "__main__":
    for filename in os.listdir(data_dir):
        data_filepath = os.path.join(data_dir, filename)
        data_df = pd.DataFrame.from_dict(get_data(data_filepath))
        data_df.to_csv(os.path.join(data_dir, filename.replace('.xml', '.csv')))
