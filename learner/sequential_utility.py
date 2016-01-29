import numpy as np
from .utility_based import UtilityBasedLearner
from utility_based import JointCheat


class Sequential(UtilityBasedLearner):
    """Class that selects snippets per document, documents are not re-ordered. These simulates a constant
    utility scenario """

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(Sequential, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                         minimax=minimax, seed=seed)

    def _subsample_pool(self, rem):
        subpool = list(rem)
        # self.rnd_state.shuffle(subpool)
        subpool = subpool[:self.subsample]

        return subpool

    def next_query(self, pool, step):

        """
        Select best candidates for asking the oracle.
        :param pool:
        :param step:
        :return:
        """
        self.subsample = step

        subpool = self._subsample_pool(pool.remaining)

        util = self.expected_utility(pool, subpool)  # util list of (utility_score, snipet index of max)

        if self.minimax > 0:  ## minimizing
            max_util = [np.argmin(p) for p in util]  # snippet per document with max/min utility
        else:
            max_util = [np.argmax(p) for p in util]  # snippet per document with max/min utility

        order = range(len(subpool)) # don't alter the order of documents

        index = [(subpool[i], max_util[i]) for i in order[:step]]

        return index


# ================================================================================================
class FirstK(Sequential):

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(FirstK, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                     seed=seed, minimax=minimax)

    def next_query(self, pool, step):

        """
        Select the first snippet of every instance in the pool.
        :param pool:
        :param step:
        :return:
        """
        self.subsample = step

        subpool = self._subsample_pool(pool.remaining)

        index = [(s, 1) for s in subpool[:step]]
        return index


#######################################################################################################################
class SequentialJointCheat(JointCheat):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1, snip_model=None):
        super(JointCheat, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                         seed=seed, minimax=minimax)
        self.snippet_model = snip_model

    def _subsample_pool(self, rem):
        subpool = list(rem)
        # self.rnd_state.shuffle(subpool)
        subpool = subpool[:self.subsample]

        return subpool

    def next_query(self, pool, step):
        self.subsample = step

        subpool = self._subsample_pool(pool.remaining)

        util = self.expected_utility(pool, subpool)  # util list of (utility_score, snipet index of max)

        if self.minimax > 0:  ## minimizing
            max_util = [np.argmin(p) for p in util]  # snippet per document with max/min utility
        else:
            max_util = [np.argmax(p) for p in util]  # snippet per document with max/min utility

        order = range(len(subpool)) # don't alter the order of documents

        index = [(subpool[i], max_util[i]) for i in order[:step]]

        return index


