"""
export_onnx.py – Export GPT PyTorch model to ONNX.

Usage (from repo root):

    python export_onnx.py \
        --checkpoint weights/best_model.pt \
        --onnx-out gpt_model.onnx

Adjust imports / config according to your repo structure.
"""

import argparse
import torch

from model import GPTModel          
from config import get_config       


def parse_args():
    parser = argparse.ArgumentParser(description="Export GPT model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained PyTorch checkpoint (.pt/.pth)",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default="gpt_model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max sequence length from config (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ----- Load config & model -----
    config = get_config()

    if args.max_seq_len is not None:
        config["max_seq_len"] = args.max_seq_len

    vocab_size = config.get("vocab_size")
    max_seq_len = config.get("max_seq_len", 128)

    if vocab_size is None:
        raise ValueError("Config must contain 'vocab_size' for dummy input generation.")

    model = GPTModel(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    # If your checkpoint has a "state_dict" key, change to state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # ----- Dummy input -----
    dummy_input = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, max_seq_len),
        dtype=torch.long,
    )

    # ----- Export -----
    print(f"Exporting GPT model to ONNX: {args.onnx_out}")
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_out,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
