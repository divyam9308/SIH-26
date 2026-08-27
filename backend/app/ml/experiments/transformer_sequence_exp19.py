"""Experiment 19: small causal Transformer encoder over monthly histories."""
from __future__ import annotations

import math
import torch
from torch import nn

from backend.app.ml.experiments.neural_sequence_common import (
    ExperimentConfig, STATIC_NUMERIC, build_embeddings, filter_comparable_rows,
    fit_sequence_experiment, predict_from_state,
)

EXPERIMENT_ID = "exp_19"
EXPERIMENT_NAME = "Small Transformer monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 19
IMPLEMENTATION_REVISION = "v1_small_transformer_history_ablation"

CONFIG = ExperimentConfig(
    experiment_id=EXPERIMENT_ID,
    experiment_name=EXPERIMENT_NAME,
    experiment_sequence=EXPERIMENT_SEQUENCE,
    implementation_revision=IMPLEMENTATION_REVISION,
    architecture_name="small_transformer",
    seed=26903,
    hypothesis="A compact self-attention encoder can model long-range interactions among monthly revisions, spending, progress and schedule signals.",
    artifact_prefix="transformer",
    selection_epochs=7,
    final_epochs=9,
)


def sinusoidal_positions(length: int, width: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    pe = torch.zeros((length, width), device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class TransformerForecaster(nn.Module):
    def __init__(self, sequence_width: int, cardinalities: list[int], hidden: int = 64):
        super().__init__()
        self.input_projection = nn.Linear(sequence_width, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=128, dropout=0.08,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden)
        self.embeddings, embedding_width = build_embeddings(cardinalities)
        static_width = len(STATIC_NUMERIC) * 2 + embedding_width
        self.static_net = nn.Sequential(nn.Linear(static_width, 32), nn.ReLU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Dropout(0.08), nn.Linear(64, 2))

    def forward(self, sequence, lengths, numeric, cats):
        batch, steps, _ = sequence.shape
        x = self.input_projection(sequence)
        x = x + sinusoidal_positions(steps, x.shape[-1], sequence.device).unsqueeze(0)
        positions = torch.arange(steps, device=sequence.device).unsqueeze(0)
        padding_mask = positions >= lengths.to(sequence.device).unsqueeze(1)
        causal_mask = torch.triu(torch.ones((steps, steps), dtype=torch.bool, device=sequence.device), diagonal=1)
        x = self.norm(self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask))
        index = torch.clamp(lengths - 1, min=0)
        temporal = x[torch.arange(batch, device=sequence.device), index]
        embeddings = [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
        static = self.static_net(torch.cat([numeric, *embeddings], dim=1))
        return self.head(torch.cat([temporal, static], dim=1))


def build_model(sequence_width: int, cardinalities: list[int]) -> nn.Module:
    return TransformerForecaster(sequence_width, cardinalities)


def fit_experiment(**kwargs):
    return fit_sequence_experiment(config=CONFIG, model_builder=build_model, **kwargs)


def predict_project(row, state):
    return predict_from_state(row, state)
