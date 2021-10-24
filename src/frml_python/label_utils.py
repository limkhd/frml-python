"""Module implementing various utils for generating labels
"""

import numpy
from collections import namedtuple
import scipy.spatial.distance as dist


def generate_training_ranking_labels_from_class(X, Y, k=None):
    """Generate for each training point a list of named tuples to indicate which
    # points should be relevant for training purposes.

    Parameters
    ----------
    X : numpy.ndarray
        n by d numpy.ndarray where n = num_examples, d = num_features

    Y : numpy.ndarray
        n by 1 numpy array where Y[i] is the integer class label of X[i]

    k : int, optional
        Number of k-nearest-neighbors of X[i] in Euclidean space to add to
        similarity set of X[i].
        If None, use all other examples X[j] in X with the same label.

    Returns
    -------
    X : numpy.ndarray

    labels: list
        List of RankingLabel namedtuples

    """

    rlabel = namedtuple("RankingLabel", "sim dif")
    labels = []

    for i in range(X.shape[0]):
        sim = numpy.where(Y == Y[i])[0]

        # if k is set, then use k nearest Euclidean neighbors as targets (like LMNN)
        if k is not None:
            sim_dist = dist.cdist([X[i]], X[sim]).flatten()
            sim = sim[numpy.argsort(sim_dist)[1 : k + 1]]

        dif = numpy.where(Y != Y[i])[0]
        labels.append(rlabel(sim, dif))

    return X, labels


def generate_test_ranking_labels_from_class(Xtest, Ytest, Xtrain, Ytrain):
    """
    Generate for each test point in Xtest a list of named tuples to indicate
    which points in Xtrain are relevant for evaluation purposes.

    Parameters
    ----------
    Xtest : numpy.ndarray
        n by d test feature matrix where n = num_examples, d = num_features

    Ytest : numpy.ndarray
        n by 1 test label array where Y[i] is the integer class label of X[i]

    Xtrain : numpy.ndarray
        n by d training feature matrix where n = num_examples, d = num_features

    Ytrain : numpy.ndarray
        n by 1 training label array where Y[i] is the integer class label of X[i]


    Returns
    -------
    Xtest : numpy.ndarray
        n by d test feature matrix where n = num_examples, d = num_features

    labels: list
        List of RankingLabel namedtuples

    """
    rlabel = namedtuple("RankingLabel", "sim dif")
    YtrainS = set(Ytrain)
    YtestS = set(Ytest)

    # Check for errors in label set

    if not YtestS.issubset(YtrainS):
        print("YTrain: %s" % repr(YtrainS))
        print("Ytest: %s" % repr(YtestS))
        raise Exception("Test labels not subset of train labels")

    labels = []

    for i in range(Xtest.shape[0]):
        sim = numpy.where(Ytrain == Ytest[i])[0]
        dif = numpy.where(Ytrain != Ytest[i])[0]
        labels.append(rlabel(sim, dif))

    return Xtest, labels
