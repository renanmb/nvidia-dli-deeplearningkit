import os
import torch
from torch.autograd import Variable
import numpy as np
import pickle
import random


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


def LoadEmbeddingsFromText(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format vector file.

    For now, values not found in the file will be set to zero."""

    emb = np.random.uniform(-0.1, 0.1, (len(vocabulary), embedding_dim), dtype=np.float32)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            if len(spl) < embedding_dim + 1:
                # Header row or final row
                continue
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:embedding_dim + 1]]
    return emb


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = ['<pad>', '<eos>', '<sos>', '<unk>']
        self.word2idx['<pad>'] = 0
        self.word2idx['<eos>'] = 1
        self.word2idx['<sos>'] = 2
        self.word2idx['<unk>'] = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, lowercase=True):
        self.dictionary = Dictionary()
        self.lowercase = lowercase
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        pickle.dump(self.dictionary, open("dictionary.p", 'wb'))


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                if self.lowercase:
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                words = ['<sos>'] + words
                words += ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            token = 0
            lines = []
            for line in f:
                if self.lowercase:
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                words = ['<sos>'] + words
                words += ['<eos>']
                lines.append([self.dictionary.word2idx[w] for w in words])

        return lines

def batchify(data, bsz, max_len=100, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []
    shortened = 0
    max_seen = 0
    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        lengths = [len(x) for x in batch]

        # Truncate
        short_batch = []
        short_lengths = []

        for i, l in enumerate(lengths):
            if l > max_len:
                short_lengths.append(max_len)
                short_batch.append(batch[i][:max_len])
                shortened += 1
                if l > max_seen:
                    max_seen = l
            else:
                short_lengths.append(l)
                short_batch.append(batch[i])

        batch = short_batch
        lengths = short_lengths

        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)

        source = [x[:-1] for x in batch]
        target = [x[1:] for x in batch]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            padding = (maxlen-len(x))*[0]
            x += padding
            y += padding
        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target))

        if gpu:
            source = source.cuda()
            target = target.cuda()
        batches.append((source, target, lengths))

    print(shortened, " shortened")
    print(max_seen, " maximum length seen")
    return batches

def truncate(indices, gpu=False, sort=True):
    """
    The purpose of this method is to truncate generated sequences
    by the first instance of the <eos> symbol (index 1)
    """

    # Assume indices Long Variables are batch_size x seq_len
    batch_size, seq_len = indices.size()

    lengths = []
    # loop through each example
    for k, ex in enumerate(indices):
        length = 0
        # loop through sequence
        for i, index in enumerate(ex):
            if index.data[0] == 1:
                # all remaining indices in sequence are 0 (padding)
                for i in range(length, seq_len):
                    ex.data[i] = 0
                    # mask[k, i] = 0
                lengths.append(length)
                break
            else:
                length += 1

        if len(lengths) <= k:
            lengths.append(length)

    if sort:
        # sort examples by length decreasing
        chunks = indices.chunk(batch_size, 0)
        indices, new_lengths = length_sort(indices, lengths)
        k = len(new_lengths)-1
        while new_lengths[k] == 0:
            indices = indices[:-1]
            new_lengths = new_lengths[:-1]
            k -= 1
        indices = torch.cat([x.unsqueeze(0) for x in indices], 0)

    else:
        new_lengths=lengths
    return to_gpu(gpu, indices),  new_lengths

def length_sort(items, lengths, descending=True):
    # must sort real and generated data by lengths
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    items = list(items)
    lengths = list(lengths)
    return items, lengths

