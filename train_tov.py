#!/usr/bin/env python3
"""
Rebel TOV Autoresearch — Phase B: Tone of Voice Training

This is the file that the agent modifies. Everything is fair game:
- Base model selection
- LoRA configuration
- Loss functions
- Training strategies

The goal: maximize tov_score and pyramid_score while minimizing perplexity.
"""

import json
import time
import sys
from pathlib import Path
from typing import Optional
import random

# === CONFIGURATION (agent can modify) ===

# Model
BASE_MODEL = "microsoft/phi-2"  # Small but capable
# Alternatives: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2-0.5B"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 512

# Loss weighting
PYRAMID_WEIGHT = 0.3  # Weight for pyramid compliance loss
TOV_WEIGHT = 0.3      # Weight for style consistency loss
LM_WEIGHT = 0.4       # Weight for language modeling loss

# Evaluation
EVAL_EVERY_STEPS = 50
TIME_BUDGET_SECONDS = 300  # 5 minutes

# === IMPORTS (do not add new packages) ===

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. Running in mock mode.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: Transformers not installed. Running in mock mode.")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("WARNING: PEFT not installed. Running in mock mode.")


# === DATA LOADING ===

class RebelTOVDataset(Dataset):
    """Dataset for TOV fine-tuning."""

    def __init__(self, corpus_path: Path, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    # Create training samples from document chunks
                    text = doc.get("text", "")
                    doc_type = doc.get("document_type", "rapport")

                    # Split into paragraphs/sections
                    paragraphs = text.split("\n\n")
                    for i, para in enumerate(paragraphs):
                        if len(para.strip()) > 100:  # Skip very short paragraphs
                            self.samples.append({
                                "text": para.strip()[:2000],
                                "doc_type": doc_type,
                                "position": "intro" if i == 0 else ("conclusion" if i == len(paragraphs)-1 else "body"),
                            })
        else:
            print(f"WARNING: Corpus not found at {corpus_path}")
            # Create mock data for testing
            self.samples = [
                {"text": f"Dit is een mock paragraaf {i} voor het testen van de pipeline.", "doc_type": "rapport", "position": "body"}
                for i in range(100)
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format as instruction
        prompt = f"Schrijf in Rebel stijl ({sample['doc_type']}, {sample['position']}):\n\n"
        full_text = prompt + sample["text"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
            "doc_type": sample["doc_type"],
        }


# === PYRAMID SCORING ===

def compute_pyramid_score(text: str) -> float:
    """
    Compute pyramid writing compliance score.

    Checks for:
    1. Top-down structure (conclusion indicators at start)
    2. MECE grouping (numbered lists, bullet points)
    3. Conciseness (sentence length)
    """
    score = 0.0

    # Check for conclusion-first patterns
    conclusion_patterns = ["concluderend", "samenvattend", "de conclusie", "kortom", "al met al"]
    first_100_chars = text[:100].lower()
    if any(p in first_100_chars for p in conclusion_patterns):
        score += 0.3

    # Check for structured elements
    if any(marker in text for marker in ["1.", "2.", "•", "-", "a)", "b)"]):
        score += 0.2

    # Check sentence length (shorter = better for pyramid)
    sentences = text.split(".")
    avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if avg_length < 20:
        score += 0.3
    elif avg_length < 30:
        score += 0.15

    # Check for evidence markers
    evidence_patterns = ["blijkt uit", "onderzoek toont", "data laat zien", "analyse wijst"]
    if any(p in text.lower() for p in evidence_patterns):
        score += 0.2

    return min(score, 1.0)


def compute_tov_score(text: str) -> float:
    """
    Compute Rebel tone of voice consistency.

    Checks for:
    1. Professional but accessible language
    2. Active voice
    3. Evidence-based statements
    """
    score = 0.5  # Baseline

    # Penalize passive voice indicators
    passive_patterns = ["wordt gedaan", "is gemaakt", "werd uitgevoerd"]
    passive_count = sum(1 for p in passive_patterns if p in text.lower())
    score -= 0.05 * passive_count

    # Reward active, direct language
    active_patterns = ["wij adviseren", "de analyse toont", "het rapport concludeert"]
    active_count = sum(1 for p in active_patterns if p in text.lower())
    score += 0.1 * active_count

    # Reward quantitative statements
    import re
    number_pattern = r'\d+[,.]?\d*\s*(%|euro|miljoen|procent)'
    numbers = re.findall(number_pattern, text.lower())
    score += 0.05 * min(len(numbers), 5)

    return max(0.0, min(score, 1.0))


# === TRAINING ===

def train():
    """Main training function."""
    start_time = time.time()

    print("=" * 60)
    print("Rebel TOV Autoresearch — Phase B: TOV Training")
    print("=" * 60)

    # Check dependencies
    if not all([HAS_TORCH, HAS_TRANSFORMERS, HAS_PEFT]):
        print("\nRunning in MOCK MODE (missing dependencies)")
        print_mock_results()
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load tokenizer
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization if GPU available
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    # Setup LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    corpus_path = Path(__file__).parent / "data" / "corpus.jsonl"
    print(f"Loading corpus: {corpus_path}")
    dataset = RebelTOVDataset(corpus_path, tokenizer, max_length=MAX_SEQ_LENGTH)
    print(f"Loaded {len(dataset)} samples")

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {TIME_BUDGET_SECONDS}s...")
    step = 0
    accumulated_loss = 0.0

    while (time.time() - start_time) < TIME_BUDGET_SECONDS:
        model.train()

        for batch in train_loader:
            if (time.time() - start_time) >= TIME_BUDGET_SECONDS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            # Evaluation
            if step % EVAL_EVERY_STEPS == 0:
                metrics = evaluate_generation(model, tokenizer, val_loader, device)
                print(f"Step {step}: tov={metrics['tov_score']:.4f}, "
                      f"pyramid={metrics['pyramid_score']:.4f}, "
                      f"ppl={metrics['perplexity']:.2f}")

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_generation(model, tokenizer, val_loader, device)

    # Get memory usage
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    # Print results
    total_time = time.time() - start_time
    print_results(final_metrics, total_time, peak_vram_mb, step)


def evaluate_generation(model, tokenizer, dataloader, device) -> dict:
    """Evaluate generation quality."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    tov_scores = []
    pyramid_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Generate samples for TOV/pyramid scoring
            generated = model.generate(
                input_ids[:, :50],  # Use first 50 tokens as prompt
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

            for gen in generated:
                text = tokenizer.decode(gen, skip_special_tokens=True)
                tov_scores.append(compute_tov_score(text))
                pyramid_scores.append(compute_pyramid_score(text))

            break  # Only evaluate on first batch for speed

    import math
    avg_loss = total_loss / max(total_samples, 1)
    perplexity = math.exp(min(avg_loss, 10))  # Cap to avoid overflow

    return {
        "tov_score": sum(tov_scores) / max(len(tov_scores), 1),
        "pyramid_score": sum(pyramid_scores) / max(len(pyramid_scores), 1),
        "perplexity": perplexity,
    }


def print_results(metrics: dict, total_time: float, peak_vram_mb: float, steps: int):
    """Print results in standard format."""
    print("\n---")
    print(f"tov_score:           {metrics['tov_score']:.6f}")
    print(f"pyramid_score:       {metrics['pyramid_score']:.6f}")
    print(f"perplexity:          {metrics['perplexity']:.2f}")
    print(f"training_seconds:    {total_time:.1f}")
    print(f"total_seconds:       {total_time:.1f}")
    print(f"peak_vram_mb:        {peak_vram_mb:.1f}")
    print(f"num_steps:           {steps}")


def print_mock_results():
    """Print mock results for testing without dependencies."""
    print("\n---")
    print(f"tov_score:           0.550000")
    print(f"pyramid_score:       0.420000")
    print(f"perplexity:          25.30")
    print(f"training_seconds:    300.0")
    print(f"total_seconds:       305.0")
    print(f"peak_vram_mb:        0.0")
    print(f"num_steps:           0")
    print("\n[MOCK MODE - install dependencies for real training]")


if __name__ == "__main__":
    train()
