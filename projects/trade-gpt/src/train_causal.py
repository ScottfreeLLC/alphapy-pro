"""
Train a causal (GPT-style) transformer for next-token prediction on price sequences.

Pure PyTorch implementation (Karpathy-inspired), predicts next bar token.

Usage:
    python -m src.train_causal --config config.yaml
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, max_length, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TradeGPT(nn.Module):
    """Causal transformer for next-token prediction on price sequences."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        self.max_length = max_length

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(hidden_dim, num_heads, max_length, dropout)
              for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_length

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.max_length:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """Simple dataset that chunks token IDs into fixed-length sequences."""

    def __init__(self, token_ids: list, block_size: int):
        self.block_size = block_size
        # Flatten all token IDs into one long sequence
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train causal GPT on price sequences")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    causal_config = config["causal"]

    tokenizer_dir = Path("data/tokenizer")
    corpus_path = Path("data/encoded/corpus.txt")
    output_dir = Path("models") / causal_config["model_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        print("ERROR: corpus.txt not found.")
        return

    # Load tokenizer
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.txt"),
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Tokenize corpus
    with open(corpus_path) as f:
        text = f.read()
    encoding = tokenizer.encode(text)
    all_ids = encoding.ids
    print(f"Corpus: {len(all_ids):,} tokens")

    # Split train/eval (90/10)
    split_idx = int(len(all_ids) * 0.9)
    train_ids = all_ids[:split_idx]
    eval_ids = all_ids[split_idx:]

    block_size = causal_config["max_length"]
    train_ds = TokenDataset(train_ids, block_size)
    eval_ds = TokenDataset(eval_ids, block_size)
    print(f"Train: {len(train_ds)} sequences, Eval: {len(eval_ds)} sequences")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model
    model = TradeGPT(
        vocab_size=vocab_size,
        hidden_dim=causal_config["hidden_dim"],
        num_layers=causal_config["num_layers"],
        num_heads=causal_config["num_heads"],
        max_length=block_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Training setup
    batch_size = causal_config["batch_size"]
    lr = causal_config["learning_rate"]
    epochs = causal_config["epochs"]
    warmup_steps = causal_config["warmup_steps"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule with warmup
    total_steps = len(train_loader) * epochs
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_eval_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Eval
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / max(1, len(eval_loader))

        print(f"Epoch {epoch}/{epochs}  train_loss={avg_train_loss:.4f}  eval_loss={avg_eval_loss:.4f}")

        # Save best
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": causal_config,
                "vocab_size": vocab_size,
                "epoch": epoch,
                "eval_loss": avg_eval_loss,
            }, output_dir / "best.pt")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": causal_config,
        "vocab_size": vocab_size,
        "epoch": epochs,
        "eval_loss": avg_eval_loss,
    }, output_dir / "final.pt")

    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Models saved to {output_dir}/")


if __name__ == "__main__":
    main()
