from typing import Any, Dict
import torch
from torch.utils.data import Dataset


def causal_mask(size: int) -> torch.Tensor:
    # Upper triangular matrix with 1s above the diagonal (future positions)
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
    return ~mask  # invert: True for allowed, False for masked


class WikiTextDataset(Dataset):
    def __init__(self, ds: Any, tokenizer: Any, seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Special token IDs (ensure these tokens are in your tokenizer)
        self.sos_id = tokenizer.token_to_id('[SOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        self.pad_id = tokenizer.token_to_id('[PAD]')

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.ds[idx]['text']
        tokens = self.tokenizer.encode(text).ids

        # Reserve spots for [SOS] and [EOS]
        max_content = self.seq_len - 2
        tokens = tokens[:max_content]
        num_pad = max_content - len(tokens)

        # Build input sequence: [SOS] + tokens + [EOS] + [PAD]*num_pad
        input_ids = torch.cat([
            torch.tensor([self.sos_id], dtype=torch.long),
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor([self.eos_id], dtype=torch.long),
            torch.full((num_pad,), self.pad_id, dtype=torch.long)
        ])  # shape: (seq_len,)

        # Build labels by shifting input_ids to the left
        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = self.pad_id

        # Attention mask to ignore padding
        attention_mask = (input_ids != self.pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)

        # Causal mask to prevent attending to future
        causal = causal_mask(self.seq_len)  # (1, seq_len, seq_len)

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'causal_mask':    causal,
            'labels':         labels,
            'text':           text
        }
