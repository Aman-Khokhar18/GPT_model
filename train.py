import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import WikiTextDataset
from model import build_gpt_model
from config import get_config, get_weight_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm


def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator((item['text'] for item in ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataloaders(config):
    # Load WikiText-2 raw splits
    train_raw = load_dataset(config['dataset_name'], config['dataset_config'], split='train')
    val_raw   = load_dataset(config['dataset_name'], config['dataset_config'], split='validation')

    # Build or load tokenizer
    tokenizer = get_or_build_tokenizer(config, train_raw)

    # Create datasets and loaders
    train_ds = WikiTextDataset(train_raw, tokenizer, config['seq_len'])
    val_ds   = WikiTextDataset(val_raw,   tokenizer, config['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader, tokenizer


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / count


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Ensure checkpoint directory
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, tokenizer = get_dataloaders(config)

    # Model
    model = build_gpt_model(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    # Tie embeddings and LM head
    model.lm_head.proj.weight = model.token_emb.embedding.weight

    # Logging
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    pad_id = tokenizer.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)

    global_step = 0
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            global_step += 1
            writer.add_scalar('train/loss', loss.item(), global_step)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f'Validation loss after epoch {epoch}: {val_loss:.4f}')
        writer.add_scalar('val/loss', val_loss, global_step)
        writer.flush()

        # Print some inference examples
        model.eval()
        console_width = 80
        num_examples = 2
        print('\nSample generations:')
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_examples:
                    break
                input_ids = batch['input_ids'].to(device)
                labels    = batch['labels'].to(device)

                logits = model(input_ids)
                preds = logits.argmax(dim=-1)

                input_text    = tokenizer.decode(input_ids[0].tolist())
                expected_text = tokenizer.decode(labels[0].tolist())
                pred_text     = tokenizer.decode(preds[0].tolist())

                print('-' * console_width)
                print(f'INPUT   : {input_text}')
                print(f'EXPECTED: {expected_text}')
                print(f'PREDICT : {pred_text}')
        print()

        # Save checkpoint
        ckpt_path = get_weight_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, ckpt_path)
        print(f'Saved checkpoint: {ckpt_path}\n')


if __name__ == '__main__':
    config = get_config()
    train_model(config)
