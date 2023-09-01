Instructions on setting up your environment can be found in the main Readme.

# SemiMNIST
Overview of Model
=================
Our model trains a single feed-forward convolutional model on 3000 labeled examples
from the MNIST dataset, but prevents overfitting by regularizing the model with
noise and by adding a reconstruction loss when running the model on unlabeled data.
Our model is inspired by Ladder Networks (Ramus et al. 2015, https://arxiv.org/abs/1507.02672;
Pezeshki et al. 2016, https://arxiv.org/pdf/1511.06430.pdf).

We first have a supervised loss based on the model's ability to correctly classify
the image. We found that utilizing data augmentation techniques (rotations and translations),
as well as by adding gaussian noise to multiple layers of the model, we were able
better prevent overfitting on the small labeled dataset.

We further regularize the model by utilizing the unlabeled data through a second
loss based on the model's ability to de-noise an image that has added gaussian noise
on multiple layers of the model. We do this by running two more copies of the model
used for supervised classification (share weights), one copy runs a "clean" version
without noise and another copy runs a "noisy" version with added gaussian noise.
We then define the reconstruction loss as the mean squared error between the
final outputs of the noisy and clean versions of the unlabeled data.
De-noising helps regularize the model, as it encourages the model to find features
that are robust to noise.

Thus within one training step, we run the same convolutional model three times
(supervised, clean unsupervised, noisy unsupervised), which gives us two loss
values that we add, and then backpropagate the error.

Requirements
============
Our model requires PyTorch 0.1.10 and scipy 0.19.0.
The pickle data files are not included in this repository as they are too large
(see https://inclass.kaggle.com/c/mnist-digit-recognition-assignment-1/data).

How to Run / Alter Code
=======================
To train a model simply run: `python mnist_model.py`

Optional Arguments
  --batch_size : for supervised training (default 50)  
  --semi-batch-size : for unsupervised training (default 50)  
  --test-batch-size : batch size for evaluation (default 1000)  
  --semi : enables use of unlabeled data for semi-supervised learning (default True)  
  --semi-weight : weight put on semi-supervised reconstruction loss (default 1)  
  --augment : boolean for whether to utilize data augmentation (translations and rotations)
              for supervised training (default True)  
  --noise : comma delimited standard deviations of gaussian noise to apply before each layer
            (default "0.3,0,0.3,0.3,0,0.3,0.3"). Note, there is one stddev than there are
            layers defined in param-file b/c noise can be appied to image (before first layer)
            and after the last layer, before the fully connected layers  
  --epochs : number of epochs to train for (default 500)  
  --lr : learning rate (default 0.001)  
  --cuda : enables cuda (default runs on cpu)  
  --seed : random seed (default 1)  
  --param-file : file to read model definition from (default model_param.txt)  
  --save : file name to save best validation model as (default "model.pt")  

The details for how to specify model structure are defined in model_param.txt.

Performance
===========
Training on default parameters on cuda yields a maximum validation accuracy of 99.41%.
