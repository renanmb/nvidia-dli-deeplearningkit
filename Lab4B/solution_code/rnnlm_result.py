import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

import data
from data import to_gpu
import models

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--load', type=str, default='ptb-1000.pt',
                    help='path to model to load; if empty str, creates new model')
parser.add_argument('--bptt', type=int, default=35,
                            help='sequence length')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, chardata, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    chardata = chardata.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    chardata = chardata.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data, chardata

eval_batch_size = 10
train_data, train_char_data = batchify(corpus.train[0], corpus.train[1], args.batch_size)
val_data, val_char_data = batchify(corpus.valid[0], corpus.valid[1], eval_batch_size)
test_data, test_char_data = batchify(corpus.test[0], corpus.test[1], eval_batch_size)
print("Data loaded")

###############################################################################
# Load the model
###############################################################################

model = torch.load(args.load, map_location=lambda storage, loc: storage)
if args.cuda:
    model.cuda()
else:
    model.cpu()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Evaluation code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    if i < 1:
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = to_gpu(args, Variable(torch.zeros(source[0:0+seq_len].size()).long(), volatile=evaluation))
        target = to_gpu(args, Variable(torch.zeros(source[0+1:0+1+seq_len].view(-1).size()).long()))
        return data, target
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens) #68259
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
