import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size, hidden_dim, num_layers=2):
        super(FeedForwardNetwork, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear((window_size - 1) * embedding_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        self.activation_function = F.relu

    def forward(self, input_sequences):
        embedded = self.embeddings(input_sequences).reshape([input_sequences.shape[0], -1])
        hidden = self.activation_function(self.hidden(embedded))
        for layer in self.hidden_layers:
            hidden = self.dropout(self.activation_function(layer(hidden)))
        output = self.output(hidden)
        return output

