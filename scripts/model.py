import torch
import torch.nn as nn
from scripts.self_attention import SelfAttention
import torch.nn.functional as F
with open('../data/train_original.txt') as f:
    lines_original = f.read().splitlines()
MAX_LENGTH = max([len(line.split()) for line in lines_original])
MAX_LENGTH = MAX_LENGTH + 1


class AttentiveEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentiveEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        linear_out = self.linear(embedded)
        output_att = linear_out.clone()
        for k in range(linear_out.shape[0]):
            output_att[k] = self.self_att(linear_out[k].unsqueeze(0), linear_out, linear_out)

        return output_att

class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)


class AttentiveEncoderPOS(nn.Module):
    def __init__(self, input_size, hidden_size, pos_size):
        super(AttentiveEncoderPOS, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(pos_size, hidden_size)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

    def forward(self, input, pos_input):
        embedded = self.embedding(input)
        pos_embedding = self.pos_embedding(pos_input)

        emb_with_pos = torch.cat([embedded, pos_embedding], dim=1)

        linear_out = self.linear(emb_with_pos)
        output_att = linear_out.clone()
        for k in range(linear_out.shape[0]):
            output_att[k] = self.self_att(linear_out[k].unsqueeze(0), linear_out, linear_out)

        return output_att