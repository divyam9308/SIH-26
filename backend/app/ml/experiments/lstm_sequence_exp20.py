"""Experiment 20: LSTM monthly sequence forecasting."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from backend.app.ml.experiments.neural_sequence_common import (
    ExperimentConfig, STATIC_NUMERIC, build_embeddings, filter_comparable_rows,
    fit_sequence_experiment, predict_from_state,
)

EXPERIMENT_ID = "exp_20"
EXPERIMENT_NAME = "LSTM monthly sequence model"
EXPERIMENT_SCOPE = "cost_delay"
EXPERIMENT_SEQUENCE = 20
IMPLEMENTATION_REVISION = "v1_lstm_history_ablation"

CONFIG = ExperimentConfig(
    experiment_id=EXPERIMENT_ID,
    experiment_name=EXPERIMENT_NAME,
    experiment_sequence=EXPERIMENT_SEQUENCE,
    implementation_revision=IMPLEMENTATION_REVISION,
    architecture_name="lstm",
    seed=27003,
    hypothesis="An LSTM memory cell may preserve useful long-horizon monthly state better than the rejected plain GRU while remaining compact enough for the available data.",
    artifact_prefix="lstm",
)


class LSTMForecaster(nn.Module):
    def __init__(self, sequence_width: int, cardinalities: list[int], hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(sequence_width, hidden, batch_first=True)
        self.embeddings, embedding_width = build_embeddings(cardinalities)
        static_width = len(STATIC_NUMERIC) * 2 + embedding_width
        self.static_net = nn.Sequential(nn.Linear(static_width, 32), nn.ReLU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(hidden + 32, 48), nn.ReLU(), nn.Dropout(0.08), nn.Linear(48, 2))

    def forward(self, sequence, lengths, numeric, cats):
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _cell) = self.lstm(packed)
        embeddings = [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
        static = self.static_net(torch.cat([numeric, *embeddings], dim=1))
        return self.head(torch.cat([hidden[-1], static], dim=1))


def build_model(sequence_width: int, cardinalities: list[int]) -> nn.Module:
    return LSTMForecaster(sequence_width, cardinalities)


def fit_experiment(**kwargs):
    return fit_sequence_experiment(config=CONFIG, model_builder=build_model, **kwargs)


def predict_project(row, state):
    return predict_from_state(row, state)
