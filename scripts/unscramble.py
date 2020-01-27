import torch
import random
import sys
sys.path.append("..")
from scripts.vocab import  Vocab, readVocab
from scripts.model import BahdanauDecoder, AttentiveEncoder
from scripts.model import SelfAttention

def load_model():
    encoder = torch.load("../models/encoder_self_attn15000.pth")
    decoder = torch.load("../models/decoder_self_attn15000.pth")
    return encoder, decoder

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


def readVocab_test():
    print("Reading lines...")

    with open('../data/test_scrambled.txt') as f:
        lines_test = f.read().splitlines()

    return lines_test
def get_train_vocab_count():
    with open('../data/train_original.txt') as f:
        lines_original = f.read().splitlines()
    MAX_LENGTH = max([len(line.split()) for line in lines_original])
    MAX_LENGTH = MAX_LENGTH + 1
    return MAX_LENGTH


def tensorFromSentence_test(vocab, sentence):
    indexes = indexesFromSentence_test(vocab, sentence)

    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def indexesFromSentence_test(vocab, sentence):
    idx = []
    for word in sentence.split(' '):
        # Ignore OOV words
        if word in vocab.word2index.keys():
            idx.append(vocab.word2index[word])

    return idx


def evaluate(encoder, decoder, sentence,vocab,  max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence_test(vocab, sentence)

        idx = input_tensor.view(input_tensor.shape[0])

        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
        final_encoder_outputs = encoder(input_tensor.view(1, input_tensor.shape[0]).squeeze(0))
        for ei in range(input_length):
            encoder_outputs[ei] = final_encoder_outputs[ei]
        SOS_token = 0
        decoder_input = torch.tensor([[SOS_token]])  # SOS

        decoder_hidden = decoder.initHidden()

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(len(input_tensor) - 1):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            temp = decoder_output.squeeze(0)
            temp = temp[idx]

            topv, topi = temp.data.topk(1)

            top_index = topi.item()
            topi = idx[topi]

            idx = torch.cat([idx[0:top_index], idx[top_index + 1:]])


            decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly_test(encoder, decoder,lines_test,ml,vocab,  n=1):
    for i in range(n):
        pair = random.choice(lines_test)
        print('>', pair)

        output_words = evaluate(encoder, decoder, pair,vocab, ml)

        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
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


def evaluate_all_test(encoder, decoder, lines_test,vocab,ml):
    unscrambled = []
    for line in lines_test:
        output_words = evaluate(encoder, decoder, line,vocab,ml)
        output_sentence = ' '.join(output_words)
        unscrambled.append(output_sentence)

    return unscrambled




def main():
    lines_test = readVocab_test()
    MAX_LENGTH = get_train_vocab_count()
    encoder, decoder = load_model()
    print("Model loaded")
    vocab, _ = prepareData()


    encoder.eval()
    decoder.eval()
    print("Predicting...")
    out = evaluate_all_test(encoder,decoder,lines_test,vocab,MAX_LENGTH)
    with open('../predictions/predictions_self_attention_v2.0_test.txt', 'w') as f:
        for item in out:
            f.write("%s\n" % item)
    print("File Saved")









if __name__ == '__main__':
    main()
