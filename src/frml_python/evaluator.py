""" Module implementing Evaluator class
"""
import numpy
import sklearn.metrics as m
import sklearn.metrics.pairwise as pd
import logging

logger = logging.getLogger(__name__)


class Evaluator(object):
    """Class to evaluate FRML models based on information retrieval metrics with the training set.

    Parameters
    ----------
    loss_weights_type : str
        Type of weighting function for WARP loss.
        Currently only accepts "rec" for reciprocal loss as described in Weston et. al.
    num_iters : int
        Number of iterations to run SGD optimizer
    stepsize : float
        Step size for stochastic gradient descent
    d : int
        Target dimension of the low-rank transformation
    batchsize : int
        Minibatch size for gradient descent. `batchsize=2` is recommended.
    report_interval : int
        Number of iterations between successive reports of validation loss
    lam : float
        Regularization parameter
    Xval : numpy.ndarray
        A :math:`n \\times input\\_dim` matrix where n is the number of validation examples
        and input_dim is the input dimensionality
    Yval : list
        A length n list of `namedtuples` where n is the number of validation examples in Xval. Each `namedtuple` associated with a validation example contains two fields `sim` and `dif` which are lists of indices of the similar and dissimilar training set examples.
    WARP_sampling_limit : int
        Maximum trials for sampling the WARP loss. See [1]
    random_state : int
        Optional random seed for reproducibility
    warm_start_L : numpy.ndarray
        Initialization matrix for `L`. If `None`, will be initialized by a random
        uniform Gaussian matrix.


    """

    def __init__(self, Xtrain, Ytrain, validation_measure="average_precision_score"):
        # can add iteration numbers for storage later
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.validation_measure = validation_measure
        pass

    def get_validation_score(self, frml_model, Xval, Yval):
        """Returns model score on validation set.

        Parameters
        ----------
        frml_model : FRML_model
            FRML_model instance that has the learnt Mahalanobis metric on the data
        Xval : numpy.ndarray
            A :math:`n \\times input\\_dim` matrix where n is the number of validation examples
            and input_dim is the input dimensionality
        Yval: numpy.ndarray
            A :math:`n_{validation} \\times 1` label vector

        Returns
        -------
        score : float


        """
        score = self.evaluate_ranking(
            frml_model,
            self.Xtrain,
            self.Ytrain,
            Xval,
            Yval,
            measure=self.validation_measure,
        )
        return score

    def generate_labels(self, Ytrain, Ytest):
        """Generates ranking labels for each point in test set from classification labels.

        Parameters
        ----------
        Ytrain : numpy.ndarray
            A :math:`n_{train} \\times 1` label vector
        Ytest: numpy.ndarray
            A :math:`n_{test} \\times 1` label vector

        Returns
        -------
        labels : numpy.ndarray
        """

        ntrain = len(Ytrain)
        ntest = len(Ytest)
        labels = numpy.zeros((ntest, ntrain))
        for i in range(ntest):
            labels[i][Ytest[i].sim] = 1
        return labels

    def evaluate_ranking(
        self,
        frml_model,
        Xtrain,
        Ytrain,
        Xtest,
        Ytest,
        measure="average_precision_score",
        k=None,
    ):
        """Evaluates test set performance of the model on a given metric and gives 
        the average result across all test samples.

        Parameters
        ----------
        frml_model : FRML_model
            FRML_model instance that has the learnt Mahalanobis metric on the data
        Xtrain : numpy.ndarray
            A :math:`n_{train} \\times d_{input}` matrix where n is the number of training 
            examples and :math:`d_{input}` is the input dimensionality
        Ytrain : numpy.ndarray
            A :math:`n_{train} \\times 1` label vector
        Xtest : numpy. ndarray
            A :math:`n_{test} \\times d_{input}` matrix where n is the number of test examples
            and :math:`d_{input}` is the input dimensionality
        Ytest: numpy.ndarray
            A :math:`n_{test} \\times 1` label vector

        Returns
        -------
        score : float
        """
        scores = -self.get_distances(frml_model, Xtrain, Xtest)
        labels = self.generate_labels(Ytrain, Ytest)

        return numpy.mean(
            [getattr(m, measure)(labels[i], scores[i]) for i, _ in enumerate(Xtest)]
        )

    def get_distances(self, frml_model, Xtrain, Xtest):
        """Calculates pairwise Mahalanobis distances between all examples of Xtrain and Xtest
        using the Mahalanobis metric given in frml_model

        Parameters
        ----------
        frml_model : FRML_model
            FRML_model instance that has the learnt Mahalanobis metric on the data
        Xtrain : numpy.ndarray
            A :math:`n_{train} \\times d_{input}` matrix where n is the number of training examples 
            and :math:`d_{input}` is the input dimensionality
        Xtest : numpy. ndarray
            A :math:`n_{test} \\times d_{input}` matrix where n is the number of test examples
            and :math:`d_{input}` is the input dimensionality

        Returns
        -------
        numpy.ndarray
            A :math:`n_{test} \\times n_{train}` matrix of distances
        """

        LXtrT = frml_model.transform(Xtrain)
        LXteT = frml_model.transform(Xtest)
        dist_matrix = pd.pairwise_distances(LXteT, LXtrT)

        return dist_matrix
