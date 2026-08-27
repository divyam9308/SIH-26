"""Experiment 17: causal Temporal Convolutional Network over monthly histories."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from backend.app.ml.experiments.neural_sequence_common import (
    ExperimentConfig, STATIC_NUMERIC, build_embeddings, filter_comparable_rows,
    fit_sequence_experiment, predict_from_state,
)

EXPERIMENT_ID = "exp_17"
EXPERIMENT_NAME = "TCN monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 17
IMPLEMENTATION_REVISION = "v1_causal_tcn_history_ablation"

CONFIG = ExperimentConfig(
    experiment_id=EXPERIMENT_ID,
    experiment_name=EXPERIMENT_NAME,
    experiment_sequence=EXPERIMENT_SEQUENCE,
    implementation_revision=IMPLEMENTATION_REVISION,
    architecture_name="causal_tcn",
    seed=26703,
    hypothesis="A causal dilated TCN can capture short- and long-range monthly trajectory motifs more reliably than recurrent compression.",
    artifact_prefix="tcn",
)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.08)

    def forward(self, x):
        residual = x
        y = self.conv(F.pad(x, (self.pad, 0)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(F.relu(y))


class TCNForecaster(nn.Module):
    def __init__(self, sequence_width: int, cardinalities: list[int], hidden: int = 48):
        super().__init__()
        self.input_projection = nn.Conv1d(sequence_width, hidden, 1)
        self.blocks = nn.ModuleList([CausalResidualBlock(hidden, d) for d in (1, 2, 4, 8, 16, 32)])
        self.embeddings, embedding_width = build_embeddings(cardinalities)
        static_width = len(STATIC_NUMERIC) * 2 + embedding_width
        self.static_net = nn.Sequential(nn.Linear(static_width, 32), nn.ReLU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(hidden + 32, 48), nn.ReLU(), nn.Dropout(0.08), nn.Linear(48, 2))

    def forward(self, sequence, lengths, numeric, cats):
        x = self.input_projection(sequence.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        temporal_all = x.transpose(1, 2)
        index = torch.clamp(lengths - 1, min=0)
        temporal = temporal_all[torch.arange(len(index), device=sequence.device), index]
        embeddings = [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
        static = self.static_net(torch.cat([numeric, *embeddings], dim=1))
        return self.head(torch.cat([temporal, static], dim=1))


def build_model(sequence_width: int, cardinalities: list[int]) -> nn.Module:
    return TCNForecaster(sequence_width, cardinalities)


def fit_experiment(**kwargs):
    return fit_sequence_experiment(config=CONFIG, model_builder=build_model, **kwargs)


def predict_project(row, state):
    return predict_from_state(row, state)
