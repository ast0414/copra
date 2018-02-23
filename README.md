# COPRA: Constrained Prominence Adversarial Attack and Defense on Sparse and Discrete Clinical Data

## Experimental Environment.
This code has been tested on Ubuntu 16.04 with Python 3.6 (by [Anaconda](https://www.anaconda.com/download/) with other packages such as numpy, scipy, and matplotlib) and [PyTorch](http://pytorch.org) 0.3.

## Step 0: Preprocess data.
First of all, we need to prepare datasets to train a model (or those used for a pre-trained model) and craft adversarial example.
We assume the following data types and structures througout the all codes used in this project.

### Features
Feature matrices are in a form of CSR (Compressed Sparse Row) matrix as defined in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and they should be saved as [pickled](https://docs.python.org/3/library/pickle.html) binary files whose filenames are `*.data_csr`.

### Labels
Labels are supposed to be pickled Python list whose filenames are `*.labels`.
For example,

```python
>>> with open("test.labels", "rb") as f:
		y_test = pickle.load(f)
>>> y_test
[0, 1, 1, 0, ..., 1, 0]
```
The `n`-th element of labels should be a label for the `n`-th row of feature CSR matrix.


## Step 1: Train a model.
Although any PyTorch model can be used in crafting step, we provide a simple example code `train_MLP.py` to train a MLP model using a class `MLP` defined in `models.py`. We assume that the dataset have already split into the three subsets; training, validation, and test set. Also, the same number of hidden units will be used in each hidden layer for simplicity.

`train_MLP.py` can be run with the following arguments:

* `data_path` (REQUIRED): a path to the directory that contains `train.data_csr, train.labels, valid.data_csr, valid.labels, test.data_csr, test.labels`.
* `--output_dir`: a path to the directory where the trained model will be saved (pickled.) Default=Current folder
* `--layers`: number of hidden layers. Default=5
* `--units`: number of hidden units in each layer. Default=256
* `--epochs`: number of epochs. Default=200
* `--batch_size`: batch size. Default=512
* `--eval_batch_size`: batch size for eval mode. Default=512
* `--lr`: initial learning rate. Default=1e-1
* `--momentum`: momentum. Default=0.9
* `--dropout`: Dropout rate at input layer. Default=0.5
* `--l1`: L1 regularization. Default=0
* `--clr`: Cross-Lipschitz regularization. Default=0
* `--no-cuda`: NOT use cuda (GPU) Default=False
* `--seed`: random seed. Default=0

For example,

```bash
$ python train_MLP Data/ --output_dir Model/ --layers 5 --units 256
```

Then, the trained model, which has 5 hidden layers each of which has 256 hidden units, will be saved at `Model/model.pth` (or `Model/model_params.pth` for their parameters only.)

## Step 2: Test the model on the clean test set.
One can easily evaluate the trained model using `test_MLP.py` with the follwing arguments:

* `model_path` (REQUIRED): a path to a trained model.
* `csr_path` (REQUIRED): a path to feature data stored in a pickled scipy CSR format.
* `label_path` (REQUIRED): a path to true labels stored in a pickled python list.
* `--eval_batch_size`: batch size for eval mode. Default=512
* `--no-cuda`: NOT use cuda (GPU) Default=False

For example,

```bash
$ python test_MLP Model/model.pth Data/test.data_csr Data/test.labels
AUROC: 0.8213584025941006, AUPR: 0.7389008123658598, F1: 0.6931479642502482, ACC: 0.7265486725663717
```

## Step 3: Craft adversarial examples.
For the main part of this project, one can craft adversarial exmaples using `craft_copra_attacks.py` with a trained model and a dataset. We use the model trained in the previous step and the test set to craft adversarial examples as an example here.

`craft_copra_attacks.py` can be executed with the following arguments:

* `model_path` (REQUIRED): a path to a trained model.
* `csr_path` (REQUIRED): a path to feature data stored in a pickled scipy CSR format.
* `label_path` (REQUIRED): a path to true labels stored in a pickled python list.
* `--output_dir`: a path to the directory where the crafted adversarial examples will be stored (pickled.) Default=Current folder
* `--max-dist`: the maximum distortion. Default=20
* `--early-stop`: Stop perturbing once the label is changed. Default=False
* `--uncon`: craft unconstrained attacks. Default=False
* `--no-cuda`: NOT use cuda (GPU) Default=False

For example,

```bash
$ python craft_copra_attacks.py Model/model.pth Data/test.data_csr Data/test.labels --output_dir Crafted/ --early-stop
Constrained Distortion 20
Crafting: 100%|█████████████████████████████████████████████████████| 1130/1130 [01:45<00:00, 10.67it/s]│
```

Then, the adversarial examples, which are crafted with at most 20 distortions and early stopping, will be stored at `Crafted/adv_samples.data_csr`

## Step 4: Test the model on the crafted adversarial examples.
One can evaluate the model with the crafted adversarial examples using the same script in Step 2:

```bash
$ python test_MLP Model/model.pth Crafted/adv_samples.data_csr Data/test.labels
AUROC: 0.21975227924884677, AUPR: 0.27051132105762565, F1: 0.15398075240594924, ACC: 0.14424778761061946
```
