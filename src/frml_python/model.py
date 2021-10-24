"""
Module implementing FRML model class
"""

import logging

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import scipy.linalg as linalg
from . import sampler
from . import evaluator

logger = logging.getLogger(__name__)
logger.debug("Loaded model module")


class FRML_model(BaseEstimator, TransformerMixin):

    """FRML model implemented as a `scikit-learn` Transformer. Based on `D. Lim,
    G. R. G. Lanckriet: Efficient Learning of Mahalanobis Metrics for Ranking,
    ICML 2014 1980-1988.`

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
        and d is the input dimensionality
    Yval : list
        A length n list of `namedtuples` where n is the number of validation
        examples in Xval. Each `namedtuple` associated with a validation example
        contains two fields `sim` and `dif` which are lists of indices of the
        similar and dissimilar training set examples.
    WARP_sampling_limit : int
        Maximum trials for sampling the WARP loss. See [1]
    random_state : int
        Optional random seed for reproducibility
    warm_start_L : numpy.ndarray
        Initialization matrix for `L`. If `None`, will be initialized by a random
        uniform Gaussian matrix.


    """

    def __init__(
        self,
        loss_weights_type="rec",
        num_iters=10000,
        stepsize=0.01,
        d=3,
        batchsize=5,
        report_interval=1000,
        lam=0.1,
        Xval=None,
        Yval=None,
        WARP_sampling_limit=5,
        random_state=None,
        warm_start_L=None,
    ):
        self.loss_weights_type = loss_weights_type
        self.num_iters = num_iters
        self.stepsize = stepsize
        self.d = d
        self.batchsize = batchsize
        self.report_interval = report_interval
        self.lam = lam
        self.Xval = Xval
        self.Yval = Yval
        self.WARP_sampling_limit = WARP_sampling_limit
        self.random_state = random_state
        if self.random_state is not None:
            logger.info(
                "Using fixed random seed of %d for this run" % self.random_state
            )

        if warm_start_L is not None and warm_start_L.shape[1] != d:
            raise Exception(
                "Dimension mismatch: L is %d, d is %d" % (warm_start_L.shape[1], d)
            )
        self.warm_start_L = warm_start_L

    def transform(self, X):
        """Transforms X to a lower dimensional space using the learned linear projection.

        Parameters
        ----------
        X : numpy.ndarray
            A :math:`n \\times d` matrix where n is the number of validation examples and d is the input dimensionality.

        Returns
        -------
        X_proj: numpy.ndarray
            A :math:`n \\times d` matrix where n is the number of validation examples and d is the target dimensionality (`self.d`)

        """
        check_is_fitted(self, "fit")

        return X.dot(self.L)

    def fit(self, X, Y):
        """Learns a linear projection from the training set that projects similar examples near to each other and dissimilar ones farther away

        Parameters
        ----------
        X : numpy.ndarray
            A :math:`n \\times d` matrix where n is the number of training examples and d is the input dimensionality.
        Y : list
            A length n list of `namedtuples` where n is the number of training examples in X. Each `namedtuple` associated with a validation example contains two fields `sim` and `dif` which are lists of indices of the similar and dissimilar training set examples.

        """

        # get seeded RNG if seed specified
        # Follows guidelines from https://scikit-learn.org/stable/developers/develop.html
        self.random_state_ = check_random_state(self.random_state)

        logger.debug("Beginning fit() method")

        self.evaluator = evaluator.Evaluator(X, Y)

        if self.warm_start_L is None:
            logger.info(
                "No warm start specified, initializing with uniform Gaussian entries"
            )
            self.L = self.random_state_.randn(X.shape[1], self.d)
        else:
            logger.info("Performing warm start from provided matrix")
            self.L = self.warm_start_L

        self.triplet_sampler = sampler.SimpleSampler(X, Y, rng=self.random_state_)
        self.num_items, self.num_features = X.shape
        self.gradient_object = {
            "grad_rank": 0,
            "p": numpy.zeros((self.batchsize * 3, self.num_features)),
            "q": numpy.zeros((self.batchsize * 3, self.num_features)),
        }

        self.samples_per_iteration = numpy.empty(self.num_iters)
        if self.loss_weights_type == "rec":
            self.loss_table = numpy.concatenate(
                (
                    numpy.array([0]),
                    numpy.cumsum(1.0 / (numpy.arange(self.num_items) + 1)),
                )
            )
        else:
            raise NotImplementedError("Invalid WARP loss weights")

        iteration_ctr = 0
        minibatch_ctr = 0
        for i in range(self.num_iters):
            iteration_ctr += 1
            (
                idx_i,
                idx_j,
                idx_k,
                violatorFound,
                trials,
                l_value,
            ) = self._get_triplet_and_loss(X, self.WARP_sampling_limit)
            x_ij = X[idx_i] - X[idx_j]
            x_ik = X[idx_i] - X[idx_k]
            minibatch_ctr += 1
            self._update_gradient_object(
                self.gradient_object, x_ij, x_ik, violatorFound, trials, l_value
            )
            logger.debug("Current grad rank: %s" % self.gradient_object["grad_rank"])

            if minibatch_ctr == self.batchsize:
                logger.debug("minibatch_ctr hit batchsize %d" % self.batchsize)
                p, q = self._get_gradient_pq_reset_object(self.gradient_object)
                self.L = self._take_gradient_step(self.L, p, q)
                minibatch_ctr = 0

                logger.debug(
                    "After reset, current grad rank: %s"
                    % self.gradient_object["grad_rank"]
                )
                logger.debug("")

            if (iteration_ctr) % self.report_interval == 0:
                self.report_validation_score(iteration_ctr)
            self.samples_per_iteration[iteration_ctr - 1] = trials

        return self

    def report_validation_score(self, iteration_ctr):
        logger.info(
            "Iteration %d/%d: Current MAP is %.3f, "
            "mean trials/iter since last report = %.3f, "
            "squared L_norm = %.3f"
            % (
                iteration_ctr,
                self.num_iters,
                self.evaluator.get_validation_score(self, self.Xval, self.Yval),
                numpy.mean(
                    self.samples_per_iteration[
                        (iteration_ctr - self.report_interval) : (iteration_ctr)
                    ]
                ),
                numpy.sum(self.L ** 2),
            )
        )

    def _get_triplet_and_loss(self, X, sampling_limit):
        idx_i, idx_j, idx_k, violatorFound, trials = self._get_warp_triplet(
            X, sampling_limit
        )

        # _get_loss_coefficient returns the coefficient of L(k), which is
        # 1/L(N) where N is negs per user
        # _get_loss_value returns estimated value of L(k), i.e. L(floor(X/N))
        l_value = self._get_loss_coefficient(
            idx_i, idx_j, trials
        ) * self._get_loss_value(idx_i, idx_j, trials)
        return (idx_i, idx_j, idx_k, violatorFound, trials, l_value)

    def _get_warp_triplet(self, X, sampling_limit):
        idx_i = self.triplet_sampler.sample_item()
        idx_j = self.triplet_sampler.sample_pos(idx_i)
        violatorFound = False
        trials = 0
        p_score = self._get_distance_score(X[idx_i], X[idx_j])
        if sampling_limit is None:
            sampling_limit = self.triplet_sampler.num_negatives_per_item[idx_i]

        while trials < sampling_limit:
            trials += 1
            idx_k = self.triplet_sampler.sample_neg(idx_i, idx_j)
            n_score = self._get_distance_score(X[idx_i], X[idx_k])
            if (p_score - n_score) < 1.0:
                violatorFound = True
                break
        return (idx_i, idx_j, idx_k, violatorFound, trials)

    def _get_loss_coefficient(self, idx_i, idx_j, trials):
        nneg = self.triplet_sampler.num_negatives_per_item[idx_i]
        return 1.0 / self.loss_table[nneg]

    def _get_loss_value(self, idx_i, idx_j, trials):
        nneg = self.triplet_sampler.num_negatives_per_item[idx_i]
        return self.loss_table[int(numpy.floor(float(nneg) / trials))]

    def _get_distance_score(self, Xi, Xj):
        """Gets Mahalanobis distance between two vectors Xi and Xj.
        (Xi-Xj)LtL(Xi-Xj)
        """

        d = Xi - Xj
        x = d.dot(self.L)
        return -x.dot(x)

    def _take_gradient_step(self, L, p, q):
        # update L with pq^T = Z
        LtL = L.T.dot(L)
        h1 = linalg.solve(LtL, p.dot(L).T)
        h2 = linalg.solve(LtL, q.dot(L).T)

        s = h1.T.dot(h2)
        h1h = L.dot(h1)
        L = L + (-0.5 * h1h + p.T + (-0.5 * p.T + 3 / 8 * h1h).dot(s)).dot(h2.T)

        return L

    def _update_gradient_object(
        self, g_obj, x_ij, x_ik, violatorFound, trials, l_value
    ):

        if violatorFound:
            logger.debug("Violator found, incrementing grad_rank by 2")
            coeff = -self.stepsize * 1.0 / self.batchsize * (1 - self.lam) * l_value

            g_obj["p"][g_obj["grad_rank"]] = coeff * x_ij
            g_obj["p"][g_obj["grad_rank"] + 1] = coeff * -x_ik

            g_obj["q"][g_obj["grad_rank"]] = x_ij
            g_obj["q"][g_obj["grad_rank"] + 1] = x_ik

            g_obj["grad_rank"] += 2

        logger.debug("Regularizer, incrementing grad_rank by 1")
        g_obj["p"][g_obj["grad_rank"]] = (
            -self.stepsize * 1.0 / self.batchsize * self.lam * x_ij
        )
        g_obj["q"][g_obj["grad_rank"]] = x_ij
        g_obj["grad_rank"] += 1

    def _get_gradient_pq_reset_object(self, gradient_object):
        grad_rank = gradient_object["grad_rank"]
        p = self.gradient_object["p"][:grad_rank]
        q = self.gradient_object["q"][:grad_rank]

        # reset object
        gradient_object["grad_rank"] = 0
        gradient_object["p"] = numpy.zeros((self.batchsize * 3, self.num_features))
        gradient_object["q"] = numpy.zeros((self.batchsize * 3, self.num_features))
        logger.debug("Resetting grad_rank to zero")

        return p, q
