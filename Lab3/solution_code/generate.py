import argparse
import time
import math
import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from utils import to_gpu, LoadEmbeddingsFromText, Corpus, batchify, truncate
from models import LanguageModel, RNNDiscriminator

parser = argparse.ArgumentParser(description='PyTorch Sequence GAN for Text')
parser.add_argument('--data', type=str, default='./snli_lm',
                    help='location of the data corpus')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--embedding_file', type=str, default='',
                    help='path to pretrained embedding file')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=50,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lowercase', type=bool,  default=True,
                    help='whether to lowercase')
parser.add_argument('--k', type=int, default=1,
                    help='number of layers')
parser.add_argument('--pretrain-lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--pretrain-epochs', type=int, default=3,
                    help='pretraining epochs limit')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--pretrain-batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--bidirectional', action='store_true',
                    help='bidirectional discriminator')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=111,
                    help='random seed')
parser.add_argument('--pretrain', action='store_true',
                    help='pretrains the generator on a language model objective')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='netG.pt',
                    help='path to save the final model')
parser.add_argument('--outf', type=str, default='./generated_output')
parser.add_argument('--netG', default='netG_best.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--sample', action='store_true',\
                    help='will sample for generation')
parser.add_argument('--temp', type=float, default=1.0,\
                    help='softmax temperaturen')
args = parser.parse_args()

with open(args.outf+'/args.txt', 'w') as f:
    f.write(str(args))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data, lowercase=args.lowercase)

eval_batch_size = 10
val_data = batchify(corpus.valid, eval_batch_size, max_len=150)
test_data = batchify(corpus.test, eval_batch_size, max_len=150)

print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)

if args.embedding_file == "":
    vocab = None
else:
    # Load pretrained word embeddings
    vocab = LoadEmbeddingsFromText(corpus.word2idx, embedding_dim=args.emsize, path=args.embedding_file)

netG = LanguageModel(rnn_type=args.rnn_type,
                     emsize=args.emsize,
                     nhidden=args.nhidden,
                     ntokens=ntokens,
                     nlayers=args.nlayers,
                     dropout=args.dropout,
                     tied=args.tied,
                     max_unroll=50,
                     initial_vocab=vocab,
                     gpu=args.cuda)
if args.netG != "":
    print('Loading generator from '+args.netG)
    netG.load_state_dict(torch.load(args.netG))

print(netG.max_unroll)

if args.cuda:
    netG.cuda()

lm_criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

netD = RNNDiscriminator(rnn_type=args.rnn_type,
                        emsize=args.emsize,
                        nhidden=args.nhidden,
                        ntokens=ntokens,
                        nlayers=args.nlayers,
                        bidirectional=args.bidirectional,
                        dropout=args.dropout,
                        initial_vocab=vocab,
                        gpu=args.cuda)
if args.netD != "":
    print('Loading discriminator from '+args.netD)
    netD.load_state_dict(torch.load(args.netD))

if args.cuda:
    netD.cuda()


# input image, noise, fixed noise for comparing learning improvement over epochs, label (real/fake)
input_ = torch.FloatTensor(args.batch_size, 9, args.emsize)
noise = torch.FloatTensor(args.nlayers, args.batch_size, args.nhidden)
fixed_noise = torch.FloatTensor(args.nlayers, args.batch_size, args.nhidden).normal_(0, 1)
label = torch.FloatTensor(args.batch_size)
ones = torch.FloatTensor(args.batch_size)
real_label = 1
fake_label = 0

input_ = to_gpu(args.cuda, Variable(input_))
label = to_gpu(args.cuda, Variable(label))
noise = to_gpu(args.cuda, Variable(noise))
fixed_noise = to_gpu(args.cuda, Variable(fixed_noise))
ones = to_gpu(args.cuda, Variable(ones))

# Binary cross entropy loss
criterion = nn.BCELoss()

if args.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input_, label = input_.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# Main Training Loop
g_losses = []
d_losses = []

netD.embedding._parameters['weight'] = netG.embedding._parameters['weight']

# do checkpointing
last_D_x = 0.5
last_D_z = None
best_test_loss = None


def lm_evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    netG.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    vocab = corpus.dictionary.idx2word
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        flat_mask = mask.view(-1, 1)
        output_mask = flat_mask.view(-1, 1).expand(flat_mask.size(0), ntokens) # examples x ntokens

        netG.zero_grad()
        output = netG(teacher_forcing=True, input_=source) # output: batch x seq_len x ntokens
        flattened_output = output.view(-1, ntokens)

        masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)

        total_loss += lm_criterion(masked_output, masked_target).data

    return total_loss[0] / len(data_source)

print("Saving generated output to "+args.outf)

for i in range(5):
    batch_size = args.batch_size

    # Resample noise for generator
    noise.data.resize_(args.nlayers, batch_size, args.nhidden)
    noise.data.normal_(0, 1)

    with open("%s/%d_no_sampling_fake.txt" % (args.outf, i), "w") as f:
        fake_embeddings, list_indices = netG(teacher_forcing=False, input_=noise, sample=False, temp=1)
        indices = torch.cat(list_indices, 1)
        fake, fake_lengths = truncate(indices, args.cuda, sort=False)
        for ex in fake.data:
            chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
            f.write(chars)
            f.write("\n")

    with open("%s/%d_sampling_fake.txt" % (args.outf, i), "w") as f:
        fake_embeddings, list_indices = netG(teacher_forcing=False, input_=noise, sample=True, temp=1)
        indices = torch.cat(list_indices, 1)
        fake, fake_lengths = truncate(indices, args.cuda, sort=False)
        for ex in fake.data:
            chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
            f.write(chars)
            f.write("\n")

    with open("%s/%d_sampling_temp0.5_fake.txt" % (args.outf, i), "w") as f:
        fake_embeddings, list_indices = netG(teacher_forcing=False, input_=noise, sample=True, temp=0.5)
        indices = torch.cat(list_indices, 1)
        fake, fake_lengths = truncate(indices, args.cuda, sort=False)
        for ex in fake.data:
            chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
            f.write(chars)
            f.write("\n")


# Run on test data.
print("Evaluating on language model task...")
test_loss = lm_evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
