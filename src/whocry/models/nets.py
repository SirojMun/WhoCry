"""Model prediksi: baseline naif, regresi linier, LSTM, dan GRU.

Semua model menerima x [B, n_in, F] dan mengeluarkan [B, n_out, n_targets]
dalam skala ternormalisasi. Empat kolom pertama fitur dinamis adalah target.
"""

import torch
from torch import nn


class Persistence(nn.Module):
    """ŷ(t+h) = y(t)."""

    def __init__(self, n_out, n_targets):
        super().__init__()
        self.n_out, self.n_targets = n_out, n_targets

    def forward(self, x):
        return x[:, -1:, : self.n_targets].expand(-1, self.n_out, -1)


class SeasonalNaive(nn.Module):
    """ŷ(t+h) = y(t+h-24); untuk h = 1..24 nilainya berada di jendela masukan."""

    def __init__(self, n_out, n_targets):
        super().__init__()
        self.n_out, self.n_targets = n_out, n_targets

    def forward(self, x):
        return x[:, -24:, : self.n_targets][:, : self.n_out]


class LinearRegression(nn.Module):
    """Regresi linier multi-output pada jendela yang diratakan.

    Dilatih dengan MSE + weight decay (setara regresi ridge) memakai loop yang sama
    dengan model lain, sehingga memori tetap kecil meskipun fiturnya banyak.
    """

    def __init__(self, n_in, n_features, n_out, n_targets):
        super().__init__()
        self.n_out, self.n_targets = n_out, n_targets
        self.linear = nn.Linear(n_in * n_features, n_out * n_targets)

    def forward(self, x):
        return self.linear(x.flatten(1)).view(-1, self.n_out, self.n_targets)


class RNNForecaster(nn.Module):
    """LSTM atau GRU; hidden state langkah terakhir -> kepala linier -> 24 × 4 keluaran."""

    def __init__(self, cell, n_features, n_out, n_targets, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.n_out, self.n_targets = n_out, n_targets
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out * n_targets)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1])).view(-1, self.n_out, self.n_targets)


def build_model(name, n_in, n_features, n_out, n_targets, **hp):
    if name == "persistence":
        return Persistence(n_out, n_targets)
    if name == "seasonal_naive":
        return SeasonalNaive(n_out, n_targets)
    if name == "linear":
        return LinearRegression(n_in, n_features, n_out, n_targets)
    if name in ("lstm", "gru"):
        return RNNForecaster(name, n_features, n_out, n_targets, **hp)
    raise ValueError(f"model tidak dikenal: {name}")


TRAINABLE = {"linear", "lstm", "gru"}
