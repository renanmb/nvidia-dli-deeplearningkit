Text Generative Adversarial Networks
====================================

## Overview

For this assignment, we wanted to explore GANs for text generation. Our main approach was to use policy-gradient methods to pass the error gradient from the discriminator to the generator. We used LSTM RNNs for both the generator and discriminator, with a shared vocabulary embedding. We pre-trained the generator with a language model objective. 

We two different policy gradient methods: REINFORCE and another method inspired by two recent papers. For REINFORCE, the reward is simply the log-likelihood of the generator tricking the discriminator. The other method is based on the works of Boundary-Seeking Generative Adversarial Networks (https://arxiv.org/pdf/1702.08431.pdf) and Maximum-Likelihood Augmented Discrete Generative Adversarial Networks (https://arxiv.org/abs/1702.07983). Their method simply reformulates the reward as to allow for more stable training. The details of this reward function is in our writeup.

We experiment primarily on the Stanford Natural Language Inference dataset sentences. We truncate sentences to 150 tokens when training and unroll for 50 words maximum (can only generate sentences with a maximum length of 50.)

We initialized the hidden states of our generator with random gaussian noise. Our hope was that the generator would learn a mapping between the noise and sentences, but most of the variation in sentence generation seems to be accounted for by the variation in sampling, as generation in autoregressive (see Softmax section in Results).

We pre-trained our generator on the language model objective and got it to a test perplexity of 28.65 (loss 3.355). With adversarial training we were only able to lower the perplexity marginally to 28.64 (loss 3.351). Overall, we believe that the results in the two papers we based our model on were only able to get 10 points in perplexity gains because their baseline pre-training generator perplexity was very high (they only trained their generator a little biton the language model objective).

You can train our model with the following command:

`python main.py --cuda --pretrain`

We also included a trained generator model with our code. To generate sentences from this model, run the following command:

`python generate.py --netG netG.pt --cuda`

## Results

We've included examples of generated sentences in the folder `generated_output`.

We found that our model was, on some occasions, able to learn to generate more reasonable and realistic sounding sentences through adversarial training.

We’re providing some examples of of sequences generated by our generator. The first line is that generated by the language model, and each successive line is generated after 500 steps of training, up to 2500 training steps. The words are generated by choosing the most likely next word one at a time (greedy decoding).

a street is a crowd .
a street corner is playing a guitar .
a street corner of a building .
a man is wearing a hat .
a street is a crowd .
a street vendor is playing a guitar .

there ’s a man in a blue shirt and a white shirt and a blue shirt is playing with a child .
there is a man in a blue shirt .
there are people .
there are a man in a blue shirt .
there are a lot of people .
there is a man in a blue shirt .

there are people in the field .
there is a man in a blue shirt and a blue shirt and a blue shirt and a blue shirt and a blue shirt and a blue shirt and a
nobody is playing with the ball .
there are people in the park .
there are people in the street .
nobody is playing a guitar .

### Softmax Temperature

We found that the temperature of the softmax used to generate words was very important in the performance of the model. Below we give examples of sequences generated at different temperatures of softmax.

All of the following sentences were generated with the same random noise, but with different temperatures.

#### Take most likely next word; No sampling

- the man is playing with a ball .
- the man is playing with a ball .
- the man is sitting on the floor . 
- there is a woman in a white shirt is sitting on the floor .
- a woman is sitting on the couch .
- the man is playing with a ball .
- a woman is sitting on the couch . 
- there is a woman in a white shirt is sitting on the floor .
- the man is playing with a ball .
- there is a woman in a white shirt is sitting on the floor .
- the man is playing with a ball . 
- the man is playing with a ball .
- a woman is sitting on the couch .
- the man is playing with a ball .
- the man is playing with a ball . 
- the man is playing with a ball .
- the man is playing with a ball .
- the man is playing with a ball .
- the man is playing with a ball .
- the man is playing with a ball . 


#### Temperature is 1.0; sampling

- a boy is riding another ricker as watch .
- the balcony is swimming .
- the son thrown street .
- four children are standing near the park in the street kitchen .
- this picture is waiting for a crowd .
- eight men are on apples
- a young woman with numbered wings , blue riding his bike
- the man is seeing sync
- the men are mid-throw in an elder .
- two girls are standing on in .
- the blond woman is broken on the beach
- two women are sitting on the side of the beach , and one behind
- senior children watch their play cement .
- the bright team in a red shirt smiles on free is at laundry .
- the lovers are asleep in the grass .
- an african girls is playing light to there is a bartender practicing violently romney is blur at the cat with hull obama .
- nobody does n’t be not other some with his dog .
- three dogs are on their bike .
- a woman and a woman elmo under behind them ’s .
- children are resting on one of the beach at a cover

#### Temperature is 0.5; sampling

- a man is standing in front of a woman in a white shirt and white shirt is walking in the street .
- the child is in the pool .
- the man in the street .
- there is a woman wearing a black shirt .
- three men are playing soccer
- the man is on the beach .
- a woman is standing with the other people .
- the man is sleeping .
- the woman is sleeping on the couch .
- two men are walking on the beach .
- the man is wearing a green shirt .
- the man is going to be on the street .
- one person is being played .
- the man is holding a red shirt .
- the girl is sitting on the floor .
- the man is in a pool .
- three girls are playing with a toy on the beach . two men are on the beach .
- the man is on the shore .
- a man is waiting for a woman


## Requirements

Our code has only been thoroughly tested with Python3.6 and PyTorch version 0.2.0_2.

## Important Files

### `main.py` - script to train model

Arguments
- data : path to data directory (with train.txt, valid.txt, and test.txt files)
- cuda : to train with GPU
- rnn_type : LSTM, GRU
- emsize : embedding size
- nhid : number of hidden units per layer
- nlayers : number of layers of recurrent net
- lr : learning rate
- clip : norm to clip (default 0.25)
- epochs : number of epochs to train for
- batch_size : batch size for training
- tied : tie word embeddigns and softmax weights (regularization technique)
- dropout : keep rate
- seed : random seed
- log-interval : interval at which to print results
- save : path + file to save model
- load : model file to load from; if empty randomly initializes weights
- eval : just evaluates the loaded model
- pretrain : flag to turn pre-training on language model objective on
- reinforce : flag to use REINFORCE reward instead of maximum likelihood augmented reward

### `generate.py` - script to load model and generate sentences
- temp : softmax temperature for generation
- netG : path to generator model file
- data : path to data directory (with train.txt, valid.txt, and test.txt files)
- cuda : use GPU

### `models.py` - contains model class definitions

### `utils.py` - contains data preprocessing code

### `A4_Furious_Logic.pdf` - our homework writeup

### `snli_lm` - contains SNLI reformatted data

### `generated_output` - contains examples of generated output