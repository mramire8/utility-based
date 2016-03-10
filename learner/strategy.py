import numpy as np
from base import Learner
from sklearn.datasets import base as bunch
from scipy.sparse import vstack
from copy import copy
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone


class RandomSampling(Learner):
    """docstring for RandomSampling"""

    def __init__(self, model):
        super(RandomSampling, self).__init__(model)


class BootstrapFromEach(Learner):
    def __init__(self, model, seed=None):
        super(BootstrapFromEach, self).__init__(model, seed=seed)
        self.rnd_state = np.random.RandomState(self.seed)

    def bootstrap(self, pool, step=2, shuffle=False):
        """
        bootstrap by selecting step/2 instances per class, in a binary dataset
        :param pool: bunch containing the available data
            pool contains:
                target: true labels of the examples
                ramaining: list of available examples in the pool to use
        :param step: how many examples to select in total
        :param shuffle: shuffle the data before selecting or not (important for sequential methods)
        :return: list of indices of selected examples
        """
        from collections import defaultdict

        step = int(step / 2)
        data = defaultdict(lambda: [])

        for i in pool.remaining:
            data[pool.target[i]].append(i)

        chosen = []
        for label in data.keys():
            candidates = data[label]
            if shuffle:
                indices = self.rnd_state.permutation(len(candidates))
            else:
                indices = range(len(candidates))
            chosen.extend([candidates[index] for index in indices[:step]])

        return chosen


class ActiveLearner(Learner):
    """ActiveLearner class that defines a simple utility based pool sampling strategy"""

    def __init__(self, model, utility=None, seed=1):
        super(ActiveLearner, self).__init__(model, seed=seed)
        self.utility = self.utility_base
        self.rnd_state = np.random.RandomState(self.seed)
        self.subsample = 100

    def utility_base(self, x):
        raise Exception("We need a utility function")


class StructuredLearner(ActiveLearner):
    """StructuredLearner is the Structured reading implementation """

    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(StructuredLearner, self).__init__(model, seed=seed)
        import copy

        self.snippet_model = copy.copy(model)
        self.utility = utility_fn
        self.snippet_utility = snippet_fn
        self.sent_tokenizer = None
        self.vct = None
        self.calibrate = None
        self.sent_rnd = np.random.RandomState(self.seed)
        self.cost_model = None
        self.cost_fn = None

    def set_cost_model(self, model):
        self.cost_model = model

    def set_cost_fn(self, fn):
        self.cost_fn = fn

    def get_name(self):
        return "{}{}".format(self.utility.__name__, self.snippet_utility.__name__)

    def fit(self, data, train=None):
        # fit student
        X = data.bow[train.index]  # vector representation
        y = train.target ## labels from the oracle
        self.model.fit(X, y)

        return self

    def _subsample_pool(self, X):
        raise NotImplementedError("Implement in the subclass")

    def set_utility(self, util):
        self.utility = getattr(self, util)

    def set_snippet_utility(self, util):
        self.snippet_utility = getattr(self, util)

    def set_sent_tokenizer(self, tokenizer):
        self.sent_tokenizer = tokenizer

    def set_vct(self, vct):
        self.vct = vct

    ## SNIPPET UTILITY FUNCTIONS

    def _snippet_rnd(self, X):
        if hasattr(X, 'shape'):
            return self.sent_rnd.random_sample(X.shape[0])
        else:
            return self.sent_rnd.random_sample(X)

    def _get_snippets(self, data, candidates):

        if hasattr(data.snippets, "shape"):
            ranges = np.cumsum(data.sizes)
            #     print 0 if i==0 else ranges[i-1],ranges[i]
            snips = []
            for i in candidates:
                snips.append(data.snippets[0 if i == 0 else ranges[i - 1]:ranges[i]])
        else:
            return [data.snippets[idx] for idx in candidates]

        return snips

    def _get_probs_per_snippet(self, data, candidates):
        ranges = np.cumsum(data.sizes)
        #     print 0 if i==0 else ranges[i-1],ranges[i]
        snips = []
        for i in candidates:
            snips.append(data.snippets[0 if i == 0 else ranges[i - 1]:ranges[i]])
        return snips

    def _compute_snippet(self, pool):

        """select the snippet with the best score for each document"""

        x_sent_bow, x_sent, x_len = self._get_snippets(pool)  # a matrix of all snippets (stacked), and the size

        snip_prob = self.snippet_model.predict_proba(x_sent_bow)

        pass

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model,
                                                                               self.snippet_model, self.utility,
                                                                               self.snippet_utility)

