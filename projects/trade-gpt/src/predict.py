"""
Inference module for trained price sequence models.

Importable by Alfi agent for real-time predictions.

Usage:
    from projects.trade_gpt.src.predict import PricePredictor
    predictor = PricePredictor.load_causal("models/trade-gpt/best.pt", "data/tokenizer")
    result = predictor.predict_next("H3P1R0V2 L1N0R1V0 H2P0R0V1")
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import torch


class PricePredictor:
    """Load a trained model and predict next bar tokens."""

    def __init__(self, model, tokenizer, device: str = "cpu", model_type: str = "causal"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type

    @classmethod
    def load_causal(cls, model_path: str, tokenizer_dir: str) -> "PricePredictor":
        """Load a trained causal GPT model."""
        from tokenizers import ByteLevelBPETokenizer
        from .train_causal import TradeGPT

        tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_dir}/vocab.json",
            f"{tokenizer_dir}/merges.txt",
        )

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        config = checkpoint["config"]

        model = TradeGPT(
            vocab_size=checkpoint["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_length=config["max_length"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return cls(model, tokenizer, device, "causal")

    @classmethod
    def load_mlm(cls, model_dir: str) -> "PricePredictor":
        """Load a trained RoBERTa MLM model."""
        from transformers import RobertaForMaskedLM, RobertaTokenizerFast

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = RobertaForMaskedLM.from_pretrained(model_dir).to(device)
        model.eval()

        return cls(model, tokenizer, device, "mlm")

    def predict_next(self, context: str, temperature: float = 0.8, top_k: int = 10) -> Dict:
        """Predict the next bar's tokens given a context string.

        Returns
        -------
        dict with keys: predicted_token, confidence, top_predictions,
                        and parsed breakdown (pivot, net, range, volume)
        """
        if self.model_type == "causal":
            return self._predict_causal(context, temperature, top_k)
        else:
            return self._predict_mlm(context)

    def _predict_causal(self, context: str, temperature: float, top_k: int) -> Dict:
        """Causal model: generate next tokens autoregressively."""
        encoding = self.tokenizer.encode(context)
        ids = encoding.ids

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(idx)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, min(5, probs.size(-1)))

        top_predictions = []
        for prob, tok_id in zip(top_probs[0].tolist(), top_ids[0].tolist()):
            decoded = self.tokenizer.decode([tok_id]).strip()
            top_predictions.append({"token": decoded, "probability": round(prob, 4)})

        best = top_predictions[0] if top_predictions else {"token": "", "probability": 0}
        parsed = self._parse_composite(best["token"])

        return {
            "predicted_token": best["token"],
            "confidence": best["probability"],
            "top_predictions": top_predictions,
            **parsed,
        }

    def _predict_mlm(self, context: str) -> Dict:
        """MLM model: fill-mask for the last position."""
        from transformers import pipeline

        fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=self.device)

        masked_context = context + " " + self.tokenizer.mask_token
        try:
            preds = fill_mask(masked_context, top_k=5)
            top_predictions = [
                {"token": p["token_str"].strip(), "probability": round(p["score"], 4)}
                for p in preds
            ]
        except Exception:
            top_predictions = []

        best = top_predictions[0] if top_predictions else {"token": "", "probability": 0}
        parsed = self._parse_composite(best["token"])

        return {
            "predicted_token": best["token"],
            "confidence": best["probability"],
            "top_predictions": top_predictions,
            **parsed,
        }

    @staticmethod
    def _parse_composite(token: str) -> Dict:
        """Parse a composite bar token like H3P1R0V2 into components."""
        parts = re.findall(r"([A-Z])(\d+)", token)
        result = {}
        type_map = {"H": "pivot", "L": "pivot", "T": "pivot",
                     "P": "net", "N": "net", "Z": "net",
                     "R": "range", "V": "volume"}
        for prefix, mag in parts:
            t = type_map.get(prefix)
            if t:
                result[t] = f"{prefix}{mag}"
        return result
