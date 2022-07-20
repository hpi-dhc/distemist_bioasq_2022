from typing import Union
from datasets import load_dataset
from pathlib import Path

data_dir = Path(__file__).parent

def load_distemist_entities():
    return load_dataset(str(data_dir / 'distemist'), 'subtrack1_entities')

def load_distemist_linking():
    return load_dataset(str(data_dir / 'distemist'), 'subtrack2_linking')