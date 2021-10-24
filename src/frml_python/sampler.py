""" Module for stochastic gradient descent sampler
"""

import logging
import numpy

logger = logging.getLogger(__name__)


class SimpleSampler(object):
    """Simple sampler to obtain training examples for stochastic gradient descent.

    Parameters
    ----------
    X : numpy.ndarray
        A :math:`n \\times input\\_dim` matrix where n is the number of training examples
        and input\\_dim is the input dimensionality.
    Y : list
        list of `RankingLabel` namedtuples defined in `frml_python.utils`. Each
        RankingLabel has `sim` and `dif` elements containing the similar and dissimilar
        examples for the corresponding training example in `X`.
    rng : numpy.random.RandomState
        RandomState used for sampling for reproducibility reasons
    """

    def __init__(self, X, Y, rng):

        self.num_items, self.num_features = X.shape
        self.X = X
        self.Y = Y
        self.rng = rng

        logger.info("Initializing sampler for SGD")

        self.num_negatives_per_item = [len(Y[i].dif) for i in range(len(Y))]

        self.item_gen = self.create_item_generator()

    def create_item_generator(self):
        """Create generator that continuously outputs item indices from a random
        permutation of the training set for stochastic gradient descent.


        Yields
        -------
        int

        """

        while True:
            s = self.rng.permutation(numpy.arange(self.num_items))

            sidx = 0
            while sidx < len(s):
                yield s[sidx]
                sidx += 1

    def sample_item(self):
        """Returns a uniformly sampled item from internal item generator.

        Returns
        -------
        int
        """

        return next(self.item_gen)

    def sample_pos(self, item):
        """Given a candidate item, returns a randomly sampled positive item for
        that candidate.

        Parameters
        ----------
        item : int

        Returns
        -------
        int

        """

        if len(self.Y[item].sim) == 0:
            return None
        else:
            pos_item = item
            while pos_item == item:
                pos_item = self.rng.choice(self.Y[item].sim)
            return pos_item

    def sample_neg(self, item, pos_item):
        """Given a candidate item, returns a randomly sampled negative item for
        that candidate.

        Currently, pos_item is not used as checking whether the sampled negative is
        a mistake is too costly and does not have much impact when there are very
        few positive items relative to the negative ones.

        Parameters
        ----------
        item : int
        pos_item: int

        Returns
        -------
        int

        """

        return self.rng.choice(self.Y[item].dif)

    def sample_triplet(self):
        """Returns a randomly sampled (item, positive, negative) tuple for stochastic
        gradient descent purposes.

        Returns
        -------
        triplet : Tuple

        """
        idx_i = self.sample_item()
        idx_j = self.sample_pos(idx_i)
        idx_k = self.sample_neg(idx_i, idx_j)
        return idx_i, idx_j, idx_k
