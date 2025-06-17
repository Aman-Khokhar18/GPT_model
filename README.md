## 🧠 GPT-from-Scratch

A minimal yet complete implementation of a GPT model in PyTorch, trained on WikiText-2 with a custom tokenizer.

---

### 📁 Repository Structure

```
GPT-from-Scratch/
├── config.py                # Configs: model, training, tokenizer, etc.
├── dataset.py              # Dataset wrapper for WikiText + tokenizer handling
├── model.py                # Transformer-based GPT architecture
├── train.py                # Training script
├── generate.py             # Text generation script (inference)
├── utils.py                # Utility functions (optional)
├── tokenizer_wikitext.json # Saved tokenizer (auto-generated)
├── weights/                # Model checkpoints saved here
├── run/                    # TensorBoard logs
├── requirements.txt        # Python dependencies
└── README.md               # Project overview & usage
```

---

### 📦 Installation

```bash
git clone https://github.com/yourusername/GPT-from-Scratch.git
cd GPT-from-Scratch
pip install -r requirements.txt
```

---

### 🧾 Requirements

```txt
torch
datasets
tokenizers
tqdm
tensorboard
```

---

### 🚀 Usage

#### 1. **Training the model**

```bash
python train.py
```

Trains a GPT model on the WikiText-2 dataset. Tokenizer is built automatically if it doesn't exist.

#### 2. **Text generation (inference)**

```bash
python generate.py
```

You’ll be prompted to enter a custom prompt, and the model will generate a continuation using the trained checkpoint.

---

### 📊 Monitoring

Track training with TensorBoard:

```bash
tensorboard --logdir=run/
```

---

### 📚 Example Output

```
Enter your prompt: The future of artificial intelligence
Generated text:
The future of artificial intelligence is uncertain, but it is likely to continue advancing in both complexity and utility...
```

---

### 🛠 Future Improvements (Suggestions)

* Add attention visualization tools
* Extend tokenizer support (BPE, SentencePiece)
* Train on larger datasets (e.g., OpenWebText)
* Add support for mixed precision training (AMP)
* Implement beam search sampling

---

### 📜 License

MIT License

---

Would you like me to generate the actual files in the right structure and upload them for easy copy-paste or GitHub setup?
