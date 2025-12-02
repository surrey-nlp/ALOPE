"""
Collators for batching QE data with multiple regression heads.
"""

from typing import Dict, Iterable, List

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


def build_head_collator(
    tokenizer: PreTrainedTokenizerBase,
    head_names: Iterable[str],
) -> DataCollatorWithPadding:
    base_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    head_names = list(head_names)

    def collate(features: List[Dict]):
        batch = base_collator(features)
        for name in head_names:
            batch[name] = torch.tensor([f.get(name, 0.0) for f in features], dtype=torch.float)
        return batch

    return collate

