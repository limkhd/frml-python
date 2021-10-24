# FRML-Python

## Overview

This is a Python implementation of the algorithm described in

Lim, D.K.H., and Lanckriet, G.R.G. Efficient Learning of Mahalanobis Metrics for
 Ranking.
Proceedings of the 31st International Conference on Machine Learning (ICML), 2014

Please cite this paper if you use this code.

## Installation

1. This demo requires Python 3.6.
2. Install the dependencies via `pip install -r requirements.txt`


## Demo Information

The included data (`data/IN300_folds.mat`) comprises 300 examples of ImageNet data from 10 classes (30 per class).
Each image is encoded as a 1000-dimensional vector using vector quantization techniques.
The complete data is split into 5 cross-validation folds with 60% training, 20% validation and 20% test data.

The demo will learn a rank-2 Mahalanobis metric between images, and will optimize it via stochastic gradient descent as in the paper.

Every 2500 iterations, the average MAP (Mean Average Precision) of the queries in the validation set is reported.

## Running the demo

From the `src` directory, run `python main.py` to run the demonstration.

## Function documentation

Sphinx documentation for module functions is available in `docs/build/index.html`
