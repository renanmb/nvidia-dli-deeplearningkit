import os
import torch


def to_gpu(cuda, var):
    if cuda:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.char2idx = {}
        self.idx2char = []

        self.char2idx = {' ':0, '{': 1, '}': 2}
        self.word2idx = {'<unk>': 0}
        self.idx2char = [' ', '{', '}']
        self.idx2word = ['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2word), len(self.idx2char)


charcounts = []
max_length = []
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        max_word_length_tmp = 0
        #with open(path, 'r', encoding="ISO-8859-1") as f:
        with open(path, 'r') as f:
            charcount = 0
            wordtokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                line = line.replace('<unk>', '|')
                line = line.replace('}', '')
                line = line.replace('{', '')
                wordtokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                    for char in word:
                        self.dictionary.add_char(char)
                    max_word_length_tmp = max(max_word_length_tmp, len(word) + 2)

        max_length.append(max_word_length_tmp)
        max_word_length = 21 # replace hardcoded value with func arg

        # Tokenize file content
        with open(path, 'r') as f:
            wordids = torch.LongTensor(wordtokens)
            charids = torch.ones(wordtokens, max_word_length).long()
            wordtoken = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    chartoken = 0
                    charids[wordtoken][chartoken] = self.dictionary.char2idx['{']

                    if word[0] == '|' and len(word) > 1:
                        word = word[2:]
                    wordids[wordtoken] = self.dictionary.word2idx[word]

                    for char in word:
                        charids[wordtoken][chartoken] = self.dictionary.char2idx[char]
                        chartoken += 1

                    charids[wordtoken][chartoken] = self.dictionary.char2idx['}']

                    wordtoken += 1

        return wordids, charids
