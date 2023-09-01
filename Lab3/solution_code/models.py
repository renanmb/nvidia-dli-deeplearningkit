import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils import to_gpu


# Language Model used for generator
class LanguageModel(nn.Module):
    def __init__(self, rnn_type, emsize, nhidden, ntokens, nlayers, dropout,
                 tied, max_unroll=30, initial_vocab=None, gpu=False):
        super(LanguageModel, self).__init__()
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.tied = tied
        self.drop = nn.Dropout(dropout)
        self.gpu = gpu
        self.max_unroll = max_unroll

        # Initialize embedding
        self.embedding = nn.Embedding(ntokens, emsize, padding_idx=0)

        # Initialize RNN cell
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size=emsize,
                                             hidden_size=nhidden,
                                             num_layers=nlayers,
                                             batch_first=True)
        else:
            raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM' and 'GRU']""")

        # Initialize Decoder Linear Transformation
        self.decoder = nn.Linear(nhidden, ntokens)
        if tied:
            if nhidden != emsize:
                raise ValueError("Cannot weights tie if nhidden != emsize")
            self.decoder.weight = self.embedding.weight

        self.init_weights(initial_vocab)


    def init_weights(self, initial_vocab):
        initrange = 0.05
        if initial_vocab is None:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        else:
            self.embedding.weight.data = torch.from_numpy(initial_vocab)
        if not self.tied:
            self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)


    def init_hidden(self, bsz):
        #weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (to_gpu(self.gpu,
                           Variable(torch.zeros(self.nlayers, bsz, self.nhidden))),
                    to_gpu(self.gpu,
                           Variable(torch.zeros(self.nlayers, bsz, self.nhidden))))
        else:
            return to_gpu(self.gpu,
                          Variable(torch.zeros(self.nlayers, bsz, self.nhidden)))


    def init_state(self, bsz):
        return to_gpu(self.gpu,
                      Variable(torch.zeros(self.nlayers, bsz, self.nhidden)))


    def forward(self, teacher_forcing, input_, sample=True, temp=1):
        # Two cases of forward pass
        # 1) Language Model Pretraining (uses teacher forcing)
        # 2) Sample Decoding (uses own output as next input)

        if teacher_forcing:
            # input_ is indices of input words
            batch_size, max_seq_len = input_.size()

            # embeddings are batch_size x max_seq_len x emsize
            embeddings = self.embedding(input_)

            # output is batch_size x max_seq_len x nhidden
            output, state = self.rnn(embeddings)
            output = self.drop(output)

            # reshape output to batch_size*max_seq_len x nhidden
            decoded = self.decoder(output.view(-1, self.nhidden))

            return decoded.view(batch_size, max_seq_len, self.ntokens)

        else:
            # input_ initial cell state noise of nlayers x batch_size x nhidden
            batch_size = input_.size(1)

            # hidden_state, cell_state
            if self.rnn_type == 'LSTM':
                state = (self.init_state(batch_size), input_)
            else:
                state = input_

            # 1 length long 'sequence' of <sos> embeddings for initial input
            start_indices = to_gpu(self.gpu,
                                   Variable(torch.zeros(batch_size, 1).fill_(2).long()))
            embedding = self.embedding(start_indices)

            all_probs = []
            all_indices = []

            for i in range(self.max_unroll):
                output, state = self.rnn(embedding, state)
                decoded = self.decoder(output.squeeze(1))

                # Want the first column to be zeros because we want 
                # the padding value to have a nonzero probability assigned to it
                probs = F.softmax(torch.div(decoded[:, 1:], temp))
                probs = torch.cat([to_gpu(self.gpu, Variable(torch.zeros(batch_size, 1))),
                                  probs], 1)

                if sample:
                    # sample from probability distribution
                    indices = probs.multinomial()
                else:
                    # greedily choose next word
                    values, indices = probs.max(1)
                    indices = indices.unsqueeze(1)

                embedding = self.embedding(indices)
                all_indices.append(indices)
                all_probs.append(probs)

            all_probs = torch.cat(all_probs, 1)

            # all_probs is batch_size x max_unroll x ntokens
            # all_indices is batch_size x max_unroll
            return all_probs, all_indices


# RNN Model for discriminator
class RNNDiscriminator(nn.Module):
    def __init__(self, rnn_type, emsize, nhidden, ntokens, nlayers, bidirectional,
                 dropout, initial_vocab=None, gpu=False):
        super(RNNDiscriminator, self).__init__()
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.drop = nn.Dropout(dropout)

        # Discriminator and Generator
        self.embedding = nn.Embedding(ntokens, emsize)

        # Discriminator
        self.rnn = nn.LSTM(input_size=emsize, hidden_size=nhidden, num_layers=nlayers,
                           dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=nhidden, out_features=1)

        self.init_weights(initial_vocab)


    def init_weights(self, initial_vocab):
        initrange = 0.05
        if initial_vocab is None:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        else:
            self.embedding.weight.data = torch.from_numpy(initial_vocab)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        if self.rnn_type == 'LSTM':
            return (to_gpu(self.gpu, Variable(torch.zeros(self.nlayers, bsz, self.nhidden))),
                to_gpu(self.gpu, Variable(torch.zeros(self.nlayers, bsz, self.nhidden))))
        else:
            return to_gpu(self.gpu, Variable(torch.zeros(self.nlayers, bsz, self.nhidden)))


    def forward(self, input_, lengths, pass_embeddings=False):
        batch_size = input_.size(0)

        if pass_embeddings:
            embeddings = input_
        else:
            embeddings = self.embedding(input_)

        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, state = self.rnn(packed_embeddings)
        if self.rnn_type == 'LSTM':
            hidden = state[0]
        else:
            hidden = state

        # hidden size is nlayers x batch_size x nhidden
        # we take the last layer of output
        hidden = self.drop(hidden[-1])
        scalar = self.linear(hidden)
        scalar = F.sigmoid(scalar.squeeze(1))
        
        return scalar


