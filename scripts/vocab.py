class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS"}
        self.n_words = 1  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class VocabPOS:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS"}
        self.n_words = 1  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readVocab():
    print("Reading lines...")

    with open('../data/train_original.txt') as f:
        lines_original = f.read().splitlines()

    with open('../data/train_scrambled.txt') as f:
        lines_scrambled = f.read().splitlines()

    pairs = list(zip(lines_scrambled, lines_original))
    vocab = Vocab('train_vocab')  # x and y vocab remains the same

    return vocab, pairs

def readVocabPOS():
    print("Reading lines...")

    with open('../data/train_original.txt') as f:
        lines_original = f.read().splitlines()

    with open('../data/train_scrambled.txt') as f:
        lines_scrambled = f.read().splitlines()

    pairs = list(zip(lines_scrambled, lines_original))
    vocab = VocabPOS('train_vocab')  # x and y vocab remains the same

    return vocab, pairs