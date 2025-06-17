from pathlib import Path


def get_config():
    return {
        # Data & tokenizer
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "seq_len": 256,
        "batch_size": 8,

        # Model architecture
        "d_model": 512,
        "num_layers": 12,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        # Optimization
        "lr": 1e-4,
        "num_epochs": 5,

        # Tokenizer
        "tokenizer_file": "tokenizer_wikitext.json",

        # Checkpointing
        "model_folder": "weights",
        "model_basename": "gpt_wikitext_",
        "preload": None,  # e.g., "3"

        # Logging
        "experiment_name": "run/gpt_wikitext"
    }


def get_weight_file_path(config: dict, epoch: int) -> str:
    folder = Path(config["model_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{config['model_basename']}{epoch}.pt"
    return str(folder / filename)
