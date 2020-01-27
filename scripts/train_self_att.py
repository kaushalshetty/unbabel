import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import time
import math

from scripts.vocab import  Vocab, readVocab
from scripts.model import BahdanauDecoder, AttentiveEncoder


def prepareData():
    vocab, pairs = readVocab()
    print("Read %s sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        vocab.addSentence(pair[0])

    print("Counted words:")
    print(vocab.name, vocab.n_words)
    return vocab, pairs
vocab, pairs = prepareData()
with open('../data/train_original.txt') as f:
    lines_original = f.read().splitlines()
MAX_LENGTH = max([len(line.split()) for line in lines_original])
MAX_LENGTH = MAX_LENGTH + 1





def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)

    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)





def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    final_encoder_outputs = encoder(input_tensor.view(1, input_tensor.shape[0]).squeeze(0))

    for ei in range(input_length):
        encoder_outputs[ei] = final_encoder_outputs[ei]
    SOS_token = 0
    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = decoder.initHidden()

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters,vocab,pairs,print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        print(iter)

        if iter % print_every == 0:
            torch.save(encoder, "../models/encoder_self_attn"+str(iter)+".pth")
            torch.save(decoder, "../models/decoder_self_attn"+str(iter)+".pth")
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))



def main():

    print(random.choice(pairs))



    hidden_size = 256
    encoder1 = AttentiveEncoder(vocab.n_words, hidden_size)
    attn_decoder1 = BahdanauDecoder(hidden_size, vocab.n_words, dropout_p=0.1)

    trainIters(encoder1, attn_decoder1, 30000,vocab,pairs, print_every=5)






if __name__ == '__main__':
    main()



