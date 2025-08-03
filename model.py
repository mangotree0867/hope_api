import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=189, cnn_filters=32, kernel_size=3, lstm_hidden_size=64, num_layers=1, num_classes=12, dropout=0.5):
        super(CNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.cnn = nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(cnn_filters)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, lengths):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = pad_packed_sequence(x, batch_first=True)
            batch_size = x.size(0)
            x = x.transpose(1, 2)
            x = x.reshape(batch_size * self.input_dim, 1, x.size(2))
            x = self.cnn(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.bn(x)
            x = self.dropout(x)
            x = x.view(batch_size, self.input_dim, self.cnn_filters, x.size(2))
            x = x.transpose(1, 2)
            x = x.reshape(batch_size, -1, self.input_dim)
        else:
            batch_size = x.size(0)
            x = x.transpose(1, 2)
            x = x.reshape(batch_size * self.input_dim, 1, x.size(2))
            x = self.cnn(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.bn(x)
            x = self.dropout(x)
            x = x.view(batch_size, self.input_dim, self.cnn_filters, x.size(2))
            x = x.transpose(1, 2)
            x = x.reshape(batch_size, -1, self.input_dim)

        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        last_indices = (lengths//2) - 1
        output = output[range(batch_size), last_indices, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output