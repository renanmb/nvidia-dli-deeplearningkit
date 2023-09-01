Instructions on setting up your environment can be found in the main Readme.

# LAB 2B

This sample code is modified from [https://github.com/pytorch/examples/tree/master/mnist](https://github.com/pytorch/examples/tree/master/mnist).

60,000 training data was split into two parts. train_labeled.p contains 3000 labeled data and train_unlabeled.p contains 57000 data without labels. They are saved as torchvision.datasets objects.
 
Loading Data:

```python
import pickle
trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
trainset_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
```

Training:
```bash
python mnist_pytorch.py
```
- Please go through and understand the sample code.
 
- Apply techniques mentioned in section 2 to improve the performance on 3000 training data samples. We are more than happy to see if you can implement other methods beyond section 2.

- Incoporate unlabeled data into this model. Try to implement some semi-supervised learning method.

Note: The sample code is tested on MacOS system, CPU, torch 0.1.7, Python 2.7.13, and Anaconda 4.3.0