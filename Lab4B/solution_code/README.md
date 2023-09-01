Language Modeling
=================

## Overview

In this assignment, we explored several different language model architectures. We tested primarily on the Penn Treebank and Gutenberg datasets. We explored many different parameters including LSTM vs. GRU, number of embedding and hidden dimensions, number of layers, learning rate, dropout, and number of training epochs among others.

We found generally that training larger models, especially with dropout, for more epochs generally lead to better performance. We found that high rates of dropout generally hurt the performance of smaller models, but effectively regularized larger models. We also found that tying the weights of the embedding and softmax weight of the decoder was a useful regularizer as well.

We also found that learning rate decay was essential to getting better performance. We started with a learning rate of 10 and reduced it by a factor of 1/4 every time the model's validation perplexity increased.

Overall our best model achieved a test perplexity of 80.31 (loss 4.39) and is reproducable withe following command: 

`python main.py --cuda --nlayers 2 --emsize 1000 --nhid 1000 --dropout 0.5 --tied --bptt 35 --lr 10`

We've included the parameters for our best PTB model with our code. You can load and evaluate our model on the PTB test dataset with the following command:

`python rnnlm_result.py --cuda --load ptb-1000.pt`

We've also included example generations from our language model in the file `generated.txt`. We also have included code to generate text using our language model. You can test out this functionality with the following command:

`python generate.py --cuda --load ptb-1000.pt`

## Requirements
Our code has only been thoroughly tested with Python3.6 and PyTorch version 0.2.0_2.

## Important files

### `main.py` - trains a language model

Arguments
- data : path to data directory (with train.txt, valid.txt, and test.txt files)
- cuda : to train with GPU
- model : LSTM, GRU
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


### `rnnlm_result.py` - evaluates language model on perplexity on test data

Arguments
- data : path to data directory (with train.txt, valid.txt, and test.txt files)
- cuda : to train with GPU
- seed : random seed
- load : file of model to load
- batch_size : batch size for evaluation


### `generate.py` - generates text from pre-trained language model

Arguments
- data : path to data directory (with train.txt, valid.txt, and test.txt files)
- cuda : to train with GPU
- seed : random seed
- load : file of model to load
- outf : file to output generated text to
- words : number of words to generate
- temperature : temperature of generation (higher will increase diversity - default 1.0)

### `models.py` - contains model class definitions

### `data.py` - contains data preprocessing code

### `Furious_Logic_A2.pdf` - contains our homework writeup

### `data` - contains Penn Treebank and Gutenberg data

