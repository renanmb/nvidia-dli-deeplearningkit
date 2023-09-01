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
parser.add_argument('--pretrain-epochs', type=int, default=1,
                    help='pretraining epochs limit')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--pretrain-batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
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
parser.add_argument('--reinforce', action='store_true',
                    help='use REINFORCE log-likelihood reward instead of Maximum Likelihood augmented reward')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='netG.pt',
                    help='path to save the final model')
parser.add_argument('--outf', type=str, default='.')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
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
# Build the generator
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
# Pretraining code
###############################################################################

def lm_train():
    # Turn on training mode which enables dropout.
    netG.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary.word2idx)
    vocab = corpus.dictionary.idx2word

    for i, batch in enumerate(train_data):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))

        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        flat_mask = mask.view(-1, 1)
        output_mask = flat_mask.view(-1, 1).expand(flat_mask.size(0), ntokens) # examples x ntokens

        netG.zero_grad()
        output = netG(teacher_forcing=True, input_=source) # output: batch x seq_len x ntokens
        flattened_output = output.view(-1, ntokens)

        masked_output = flattened_output.masked_select(output_mask).view(-1, ntokens)
        loss = lm_criterion(masked_output, masked_target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(netG.parameters(), args.clip)
        for p in netG.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_data), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    return total_loss


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


# Loop over epochs.
lr = args.pretrain_lr
best_val_loss = None
stats = []

if args.pretrain:
    print("Pretraining generator with language model objective...")
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.pretrain_epochs+1):
            train_data = batchify(corpus.train, args.pretrain_batch_size, shuffle=True, max_len=120)
            epoch_start_time = time.time()
            train_loss = lm_train()
            val_loss = lm_evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            stats.append((epoch, train_loss, val_loss, math.exp(val_loss)))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                print("Saving with best val loss", val_loss)
                with open(args.save, 'wb') as f:
                    torch.save(netG.state_dict(), f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr *= 0.25
            pickle.dump(stats, open("stats_"+args.save[:-3]+".p", "wb"))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

        # Load the best saved model.
        with open(args.save, 'rb') as f:
            netG.load_state_dict(torch.load(f))
        # Run on test data.
        test_loss = lm_evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
else:
    print("Not pretraining generator with language model objective")


###############################################################################
# Build the discriminator
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

###############################################################################
# Training setup code
###############################################################################

# Setup Optimizers
optimizerD = optim.SGD(netD.parameters(), lr=args.lr)
# optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.SGD(netG.parameters(), lr=args.lr)
# optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

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

# Discriminator and Generator share embedding weights
netD.embedding._parameters['weight'] = netG.embedding._parameters['weight']

# do checkpointing
torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, 0))
torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, 0))
last_D_x = 0.5
last_D_z = None
best_test_loss = None

train_data = batchify(corpus.train, args.batch_size, shuffle=True, max_len=50)

best_G = '%s/netG_epoch_%d.pth' % (args.outf, 0)

###############################################################################
# Language Model Generation
###############################################################################

print("\nLanguage Model Generation")
# Generate fake sentences
fake_probs, list_indices = netG(teacher_forcing=False, input_=fixed_noise,
                                sample=args.sample, temp=args.temp)
indices = torch.cat(list_indices, 1)

with open("%s/language_model.txt" % args.outf, "w") as f:
    fake, fake_lengths = truncate(indices, args.cuda, sort=False)
    for ex in fake.data:
        chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
        f.write(chars)
        f.write("\n")

###############################################################################
# Training loop
###############################################################################

# At any point you can hit Ctrl + C to break out of training early.
try:
    print("\nTraining...")
    # Loop through epochs
    for epoch in range(1, args.epochs+1):

        # Loop through batches
        for i, (data, target, lengths) in enumerate(train_data):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            data = data[:, 1:]
            netD.zero_grad()
            batch_size = data.size(0)

            # Train with Real Examples ----------------------------
            label.data.resize_(batch_size).fill_(real_label)
            data = to_gpu(args.cuda, Variable(data))

            output = netD(data, lengths)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # Train with Generated Examples ----------------------------

            # Resample noise for generator
            noise.data.resize_(args.nlayers, batch_size, args.nhidden)
            noise.data.normal_(0, 1)

            # Generate fake text
            fake_probs, list_indices = netG(teacher_forcing=False, input_=noise, sample=True, temp=args.temp)
            indices = torch.cat(list_indices, 1)

            fake, fake_lengths = truncate(indices, args.cuda)
            label.data.resize_(len(fake_lengths)).fill_(fake_label)

            # detach b/c discriminator does NOT backpropagate into the generator (adversarial objective)
            output = netD(fake.detach(), fake_lengths)
            errD_fake = criterion(output, label)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.data.mean()

            errD = errD_real + errD_fake

            if last_D_z is None:
                last_D_z = D_G_z1

            if D_x < 0.9: 
                # Sum losses from training on real and fake examples
                optimizerD.step()
                last_D_x = D_x
                last_D_z = D_G_z1

            netG.embedding._parameters['weight'] = netD.embedding._parameters['weight']
            d_losses.append((epoch, i, errD.data.cpu()[0]))

            if epoch != 1 or i > args.k:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.data.fill_(real_label)  # fake labels are real for generator cost
                # generator gets signal from discriminator, don't detach output
                prob_real = netD(fake, fake_lengths)
                generated_lengths = to_gpu(args.cuda, Variable(torch.from_numpy(np.array(fake_lengths))))

                # Regular REINFORCE: reward is log probability of objective (tricking discriminator)
                #log_prob = prob_real.log()

                # Maximum-Likelihood Augmented
                ones.data.resize_(len(fake_lengths)).fill_(1)
                weight = torch.div(prob_real, ones - prob_real)
                total = torch.sum(weight)
                mlreward = torch.div(weight, total.data[0])

                if mlreward.size(0) < batch_size:
                    ones.data.resize_(batch_size-mlreward.size(0)).fill_(0)
                    mlreward = torch.cat([mlreward, ones], 0)

                # Policy gradient reward
                total_reward = 0
                for j, action in enumerate(list_indices):
                    # Create sentence length mask over padding
                    mask = generated_lengths.gt(j).float()
                    if mask.size(0) < batch_size:
                        mask = torch.cat([mask, ones], 0)
                    if args.reinforce:
                        reward = log_prob * mask
                    else:
                        reward = mlreward * mask
                    reward = reward.unsqueeze(1) 
                    action.reinforce(reward.data)
                    total_reward += torch.sum(reward)

                optimizerG.zero_grad()
                autograd.backward(list_indices, [None for _ in list_indices], retain_graph=True)
                optimizerG.step()

                total_reward = total_reward.data.cpu()[0] / batch_size
                g_losses.append((epoch, i, total_reward))
                netD.embedding._parameters['weight'] = netG.embedding._parameters['weight']

                if i % 25 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                        % (epoch, args.epochs, i, len(train_data),
                            errD.data[0], total_reward, D_x, D_G_z1))

                    with open("%s/%d-%d-epoch_real.txt" % (args.outf, i, epoch), "w") as f:
                        indices = data.data.cpu().numpy()
                        for ex in indices:
                            chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
                            f.write(chars)
                            f.write("\n")

                    with open("%s/%d-%d-epoch_fake.txt" % (args.outf, i, epoch), "w") as f:
                        fake_embeddings, list_indices = netG(teacher_forcing=False,
                                                             input_=fixed_noise,
                                                             sample=args.sample,
                                                             temp=args.temp)
                        indices = torch.cat(list_indices, 1)
                        fake, fake_lengths = truncate(indices, args.cuda, sort=False)
                        for ex in fake.data:
                            chars = " ".join([corpus.dictionary.idx2word[x] for x in ex])
                            f.write(chars)
                            f.write("\n")

            if i % 100 == 0:
                # Run on test data.
                test_loss = lm_evaluate(test_data)
                print('=' * 89)
                print('| test loss {:5.2f} | test ppl {:8.2f}'.format(
                    test_loss, math.exp(test_loss)))
                print('=' * 89)

                if best_test_loss is None or best_test_loss > test_loss:
                    print("saving with new best loss", test_loss)
                    # do checkpointing
                    best_G = '%s/netG_epoch_%d_best.pth' % (args.outf, epoch)
                    torch.save(netG.state_dict(), '%s/netG_epoch_%d_best.pth' % (args.outf, epoch))
                    torch.save(netD.state_dict(), '%s/netD_epoch_%d_best.pth' % (args.outf, epoch))
                    pickle.dump(g_losses, open("%s/netG_stats_epoch_%d_best.p" % (args.outf, epoch), "wb"))
                    pickle.dump(d_losses, open("%s/netD_stats_epoch_%d_best.p" % (args.outf, epoch), "wb"))
                    best_test_loss = test_loss

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        pickle.dump(g_losses, open("%s/netG_stats_epoch_%d.p" % (args.outf, epoch), "wb"))
        pickle.dump(d_losses, open("%s/netD_stats_epoch_%d.p" % (args.outf, epoch), "wb"))


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

    # # Load the best saved model.
    with open(best_G, 'rb') as f:
        netG.load_state_dict(torch.load(f))

    # Run on test data.
    test_loss = lm_evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    exit(0)

print('-' * 89)
print('Exiting from training early')

# Load the best saved model.
with open(best_G, 'rb') as f:
    netG.load_state_dict(torch.load(f))

# Run on test data.
test_loss = lm_evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
