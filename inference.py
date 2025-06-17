import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from config import get_config
from model import build_gpt_model

from pathlib import Path

def load_model(config, tokenizer, device):
    model = build_gpt_model(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    
    # Load the trained weights
    checkpoint = torch.load(Path(config['model_folder']) / f"{config['model_basename']}{config['preload']}.pt", 
                           map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Tie embeddings and LM head
    model.lm_head.proj.weight = model.token_emb.embedding.weight
    
    return model

def load_tokenizer(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))

def generate_text(model, tokenizer, prompt, max_length, temperature, device):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [tokenizer.token_to_id('[SOS]')] + input_ids
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_ids = input_ids
    
    for _ in range(max_length):
        # Get model predictions
        with torch.no_grad():
            logits = model(generated_ids)
        
        # Focus on the last token
        logits = logits[:, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to the generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        # Stop if we reach EOS token
        if next_token.item() == tokenizer.token_to_id('[EOS]'):
            break
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    # Remove SOS/EOS tokens for cleaner output
    generated_text = generated_text.replace('[SOS]', '').replace('[EOS]', '').strip()
    return generated_text

def main():
    # Load config
    config = get_config()
    config['preload'] = '5'  # Load the final epoch model by default
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(config)
    model = load_model(config, tokenizer, device)
    
    print("\nGPT Text Generation (type 'exit' to quit)")
    print("--------------------------------------")
    
    while True:
        # Get user input
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() == 'exit':
            break
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=100,  # Maximum tokens to generate
            temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = more random)
            device=device
        )
        
        print("\nGenerated text:")
        print(generated_text)

if __name__ == '__main__':
    main()