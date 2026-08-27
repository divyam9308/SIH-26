"""Experiment 18: GRU with learned temporal attention over monthly histories."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from backend.app.ml.experiments.neural_sequence_common import (
    ExperimentConfig, STATIC_NUMERIC, build_embeddings, filter_comparable_rows,
    fit_sequence_experiment, predict_from_state,
)

EXPERIMENT_ID = "exp_18"
EXPERIMENT_NAME = "GRU attention monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 18
IMPLEMENTATION_REVISION = "v1_gru_attention_history_ablation"

CONFIG = ExperimentConfig(
    experiment_id=EXPERIMENT_ID,
    experiment_name=EXPERIMENT_NAME,
    experiment_sequence=EXPERIMENT_SEQUENCE,
    implementation_revision=IMPLEMENTATION_REVISION,
    architecture_name="gru_attention",
    seed=26803,
    hypothesis="Temporal attention can let a GRU emphasize historically important revisions or deterioration episodes instead of relying only on the final hidden state.",
    artifact_prefix="gru_attention",
)


class GRUAttentionForecaster(nn.Module):
    def __init__(self, sequence_width: int, cardinalities: list[int], hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(sequence_width, hidden, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.embeddings, embedding_width = build_embeddings(cardinalities)
        static_width = len(STATIC_NUMERIC) * 2 + embedding_width
        self.static_net = nn.Sequential(nn.Linear(static_width, 32), nn.ReLU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(hidden + 32, 48), nn.ReLU(), nn.Dropout(0.08), nn.Linear(48, 2))

    def forward(self, sequence, lengths, numeric, cats):
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=sequence.shape[1])
        scores = self.attention(outputs).squeeze(-1)
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        mask = positions >= lengths.to(sequence.device).unsqueeze(1)
        weights = torch.softmax(scores.masked_fill(mask, -1e9), dim=1)
        temporal = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        embeddings = [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
        static = self.static_net(torch.cat([numeric, *embeddings], dim=1))
        return self.head(torch.cat([temporal, static], dim=1))


def build_model(sequence_width: int, cardinalities: list[int]) -> nn.Module:
    return GRUAttentionForecaster(sequence_width, cardinalities)


def fit_experiment(**kwargs):
    return fit_sequence_experiment(config=CONFIG, model_builder=build_model, **kwargs)


def predict_project(row, state):
    return predict_from_state(row, state)
