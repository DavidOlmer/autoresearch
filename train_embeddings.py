#!/usr/bin/env python3
"""
Rebel TOV Autoresearch — Phase A: Embedding Training

This is the file that the agent modifies. Everything is fair game:
- Model architecture
- Loss functions
- Hyperparameters
- Pooling strategies
- Training loop

The goal: maximize cluster_silhouette and pyramid_correlation.
"""

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# === CONFIGURATION (agent can modify) ===

# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
POOLING_STRATEGY = "mean"  # Options: mean, cls, max, attention

# Training
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 256

# Loss
LOSS_TYPE = "contrastive"  # Options: contrastive, triplet, multiple_negatives
TEMPERATURE = 0.05
HARD_NEGATIVE_RATIO = 0.5

# Weak supervision
FILENAME_NOISE_TOKENS = {
    "clean",
    "comments",
    "concept",
    "conceptversie",
    "cs",
    "def",
    "definitief",
    "definitieve",
    "draft",
    "eind",
    "eo",
    "final",
    "finale",
    "lf",
    "opmerkingen",
    "reacties",
    "rebel",
    "rep",
    "review",
    "schoon",
    "trackchanges",
}

# Evaluation
EVAL_EVERY_STEPS = 100
TIME_BUDGET_SECONDS = 300  # 5 minutes

# === IMPORTS (do not add new packages) ===

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except (ImportError, OSError) as e:
    HAS_TORCH = False
    print(f"WARNING: PyTorch not available ({type(e).__name__}). Running in mock mode.")
    # Mock classes for testing utility functions without torch
    class Dataset:
        pass
    class DataLoader:
        pass
    class nn:
        class Module:
            pass
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: Transformers not installed. Running in mock mode.")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.stats import pearsonr
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not installed. Running in mock mode.")


# === DATA LOADING ===


def infer_document_label(document: dict) -> str:
    """
    Infer a weak supervision label from the filename.

    The extracted corpus does not currently expose project metadata, so we
    derive a stable document-family label from the filename by removing dates,
    version markers, and review-state suffixes.
    """
    filename = str(document.get("filename", "")).strip()
    if not filename:
        return f"document {document.get('id', 'unknown')}"

    stem = Path(filename).stem.lower()
    stem = re.sub(r"^[\W_]*\d{6,8}[\W_]*", "", stem)
    # Remove version patterns before space normalization (e.g., v1.6, _v2.0_)
    stem = re.sub(r"[_\s]*v\d+(?:\.\d+)*[_\s]*", " ", stem)
    stem = re.sub(r"[\W_]+", " ", stem)

    filtered_tokens = []
    for token in stem.split():
        if re.fullmatch(r"\d{4,8}", token):  # Filter date-like numbers (4-8 digits)
            continue
        if token in FILENAME_NOISE_TOKENS:
            continue
        filtered_tokens.append(token)

    label = " ".join(filtered_tokens).strip()
    if not label:
        return f"document {document.get('id', 'unknown')}"
    return label


def build_document_preview(text: str, max_characters: int = 4000) -> str:
    """
    Keep signal-rich headings before truncating the document body.

    Rebel reports are long and front-loaded with structure. Preserving
    headings such as management summary, conclusions, and recommendations
    yields a more representative preview than raw character truncation.
    """
    if not text:
        return ""

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return ""

    key_sections = []
    for line in stripped_lines:
        line_lower = line.lower()
        is_short_heading = len(line.split()) <= 8 and len(line) <= 80
        has_priority_keyword = any(
            keyword in line_lower
            for keyword in [
                "managementsamenvatting",
                "samenvatting",
                "conclus",
                "aanbevel",
                "inleiding",
                "doel",
            ]
        )
        if is_short_heading or has_priority_keyword:
            if line not in key_sections:
                key_sections.append(line)

    ordered_sections = key_sections + stripped_lines[:20]
    preview_parts = []
    current_length = 0
    for line in ordered_sections:
        addition = len(line) + (1 if preview_parts else 0)
        if current_length + addition > max_characters:
            break
        preview_parts.append(line)
        current_length += addition

    return "\n".join(preview_parts)


def encode_string_labels(values: list[str], device) -> "torch.Tensor":
    """Map string labels to stable integer IDs within a batch."""
    unique_values = sorted(set(values))
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    return torch.tensor([value_to_index[value] for value in values], device=device)


class RebelCorpusDataset(Dataset):
    """Dataset for Rebel documents."""

    def __init__(self, corpus_path: Path, tokenizer, max_length: int = 256):
        self.documents = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    self.documents.append(doc)
        else:
            print(f"WARNING: Corpus not found at {corpus_path}")
            # Create mock data for testing
            self.documents = [
                {"id": f"mock_{i}", "text": f"Mock document {i} for testing.", "project_number": f"P{i % 10}"}
                for i in range(100)
            ]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        text = build_document_preview(doc.get("text", ""), max_characters=4000)
        label = infer_document_label(doc)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "project": label,
            "doc_id": doc.get("id", str(idx)),
        }


# === MODEL ===

class EmbeddingModel(nn.Module):
    """Sentence embedding model with configurable pooling."""

    def __init__(self, model_name: str, pooling: str = "mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "mean":
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        elif self.pooling == "cls":
            # CLS token
            embeddings = hidden_states[:, 0, :]

        elif self.pooling == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9
            embeddings = torch.max(hidden_states, dim=1)[0]

        else:
            # Default to mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        return F.normalize(embeddings, p=2, dim=1)


# === LOSS FUNCTIONS ===

class ContrastiveLoss(nn.Module):
    """Contrastive loss with in-batch negatives."""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create label mask (1 if same project, 0 otherwise)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Remove diagonal
        mask = torch.eye(similarity.size(0), device=similarity.device).bool()
        similarity = similarity.masked_fill(mask, -1e9)
        label_matrix = label_matrix.masked_fill(mask, 0)

        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean log probability for positive pairs
        pos_mask = label_matrix > 0
        if pos_mask.sum() > 0:
            loss = -(log_prob * pos_mask).sum() / pos_mask.sum()
        else:
            loss = torch.tensor(0.0, device=similarity.device)

        return loss


# === EVALUATION ===

def evaluate_embeddings(model, dataloader, device) -> dict:
    """Evaluate embedding quality."""
    model.eval()

    all_embeddings = []
    all_projects = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            all_projects.extend(batch["project"])

    import numpy as np
    embeddings_np = np.vstack(all_embeddings)

    # Convert projects to numeric labels
    unique_projects = list(set(all_projects))
    project_to_idx = {p: i for i, p in enumerate(unique_projects)}
    labels = np.array([project_to_idx[p] for p in all_projects])

    # Clustering evaluation
    n_clusters = min(len(unique_projects), 10)
    if len(embeddings_np) > n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        silhouette = silhouette_score(embeddings_np, cluster_labels)
    else:
        silhouette = 0.0

    # Pyramid correlation (mock - would need actual pyramid scores)
    # For now, use random correlation as placeholder
    pyramid_scores = np.random.random(len(embeddings_np))
    embedding_norms = np.linalg.norm(embeddings_np, axis=1)
    correlation, _ = pearsonr(embedding_norms, pyramid_scores)

    return {
        "cluster_silhouette": silhouette,
        "pyramid_correlation": abs(correlation),
        "n_documents": len(embeddings_np),
        "n_projects": len(unique_projects),
    }


# === TRAINING LOOP ===

def train():
    """Main training function."""
    start_time = time.time()

    print("=" * 60)
    print("Rebel TOV Autoresearch — Phase A: Embeddings")
    print("=" * 60)

    # Check dependencies
    if not all([HAS_TORCH, HAS_TRANSFORMERS, HAS_SKLEARN]):
        print("\nRunning in MOCK MODE (missing dependencies)")
        print_mock_results()
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load tokenizer and model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmbeddingModel(MODEL_NAME, pooling=POOLING_STRATEGY).to(device)

    # Load data
    corpus_path = Path(__file__).parent / "data" / "corpus.jsonl"
    print(f"Loading corpus: {corpus_path}")
    dataset = RebelCorpusDataset(corpus_path, tokenizer, max_length=MAX_SEQ_LENGTH)
    print(f"Loaded {len(dataset)} documents")

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
    loss_fn = ContrastiveLoss(temperature=TEMPERATURE)

    # Training loop
    print(f"\nTraining for {TIME_BUDGET_SECONDS}s...")
    step = 0
    best_silhouette = 0.0

    while (time.time() - start_time) < TIME_BUDGET_SECONDS:
        model.train()

        for batch in train_loader:
            if (time.time() - start_time) >= TIME_BUDGET_SECONDS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            projects = batch["project"]

            # Convert weak supervision labels to tensor IDs for the batch.
            labels = encode_string_labels(list(projects), device=device)

            # Forward pass
            embeddings = model(input_ids, attention_mask)
            loss = loss_fn(embeddings, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # Evaluation
            if step % EVAL_EVERY_STEPS == 0:
                metrics = evaluate_embeddings(model, val_loader, device)
                print(f"Step {step}: silhouette={metrics['cluster_silhouette']:.4f}, "
                      f"pyramid_corr={metrics['pyramid_correlation']:.4f}")

                if metrics["cluster_silhouette"] > best_silhouette:
                    best_silhouette = metrics["cluster_silhouette"]

    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_embeddings(model, val_loader, device)

    # Get memory usage
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_vram_mb = 0.0

    # Print results
    total_time = time.time() - start_time
    print_results(final_metrics, total_time, peak_vram_mb, step)


def print_results(metrics: dict, total_time: float, peak_vram_mb: float, steps: int):
    """Print results in standard format."""
    print("\n---")
    print(f"cluster_silhouette:  {metrics['cluster_silhouette']:.6f}")
    print(f"pyramid_correlation: {metrics['pyramid_correlation']:.6f}")
    print(f"training_seconds:    {total_time:.1f}")
    print(f"total_seconds:       {total_time:.1f}")
    print(f"peak_vram_mb:        {peak_vram_mb:.1f}")
    print(f"num_steps:           {steps}")
    print(f"n_documents:         {metrics['n_documents']}")
    print(f"n_projects:          {metrics['n_projects']}")


def print_mock_results():
    """Print mock results for testing without dependencies."""
    print("\n---")
    print(f"cluster_silhouette:  0.250000")
    print(f"pyramid_correlation: 0.450000")
    print(f"training_seconds:    300.0")
    print(f"total_seconds:       305.0")
    print(f"peak_vram_mb:        0.0")
    print(f"num_steps:           0")
    print(f"n_documents:         100")
    print(f"n_projects:          10")
    print("\n[MOCK MODE - install dependencies for real training]")


if __name__ == "__main__":
    train()
