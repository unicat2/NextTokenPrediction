import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.5):

        super(LSTMModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(inputs)
        output, (hidden, cell) = self.lstm(embeds)
        output = self.linear(output)
        return output


