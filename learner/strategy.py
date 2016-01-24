import numpy as np
from base import Learner
from sklearn.datasets import base as bunch
from scipy.sparse import vstack
from copy import copy


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
        return self.sent_rnd.random_sample(X.shape[0])

    def _get_snippets(self, data, candidates):
        ranges = np.cumsum(data.sizes)
        #     print 0 if i==0 else ranges[i-1],ranges[i]
        snips = []
        for i in candidates:
            snips.append(data.snippets[0 if i == 0 else ranges[i - 1]:ranges[i]])
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


class Joint(StructuredLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.current_training = []
        self.current_training_labels = []
        self.minimax = minimax
        self.validation_index = []
        self.loss_fn = None

    def set_minimax(self, minimax):
        if minimax == 'maximize':
            self.minimax = -1
        else:
            self.minimax = 1

    def _subsample_pool(self, rem):
        subpool = list(rem)
        self.rnd_state.shuffle(subpool)
        subpool = subpool[:250]

        return subpool

    def next_query(self, pool, step):

        """
        Select best candidates for asking the oracle.
        :param pool:
        :param step:
        :return:
        """
        subpool = self._subsample_pool(pool.remaining)

        util = self.expected_utility(pool, subpool)  # util list of (utility_score, snipet index of max)

        if self.minimax > 0:  ## minimizing
            max_util = [np.argmin(p) for p in util]  # snippet per document with max/min utility
        else:
            max_util = [np.argmax(p) for p in util]  # snippet per document with max/min utility

        order = np.argsort([util[i][u] for i, u in enumerate(max_util)])[
                ::self.minimax]  # document with snippet utlity max/min

        index = [(subpool[i], max_util[i]) for i in order[:step]]

        return index

    def expected_utility(self, data, candidates):
        labels = self.model.classes_
        tra_y = copy(self.current_training_labels)
        tra_x = copy(self.current_training)
        clf = copy(self.model)
        utilities = []  # two per document
        for i, x_i in enumerate(candidates):
            tra_x.append(x_i)
            uts = []

            for lbl in labels:
                tra_y.append(lbl)
                clf.fit(data.bow[tra_x], tra_y)
                u = self.evaluation_on_validation(clf, data.bow[data.validation], data.target[data.validation])
                uts.append((i, lbl, u))
                # undo labels
                tra_y = tra_y[:-1]

            utilities.append(uts)

            # undo training instance
            tra_x = tra_x[:-1]

        snippets = self._get_snippets(data, candidates)
        probs = [self.snippet_model.predict_proba(snip) for snip in snippets]  # one per snippet
        cost = data.snippet_cost[candidates]
        exp_util = []

        for ut, pr, co in zip(utilities, probs, cost):  # for every document
            exp = []
            for p, c in zip(pr, co):  # for every snippet in the document
                exp.append((p[0] * ut[0][2] + p[1] * ut[1][2]) / c)  ## utility over cost

            exp_util.extend([exp])  # one per snippet

        return exp_util

    def evaluation_on_validation(self, clf, data, target):

        loss = self.loss_fn(clf, data, target)

        return loss

    def fit(self, data, train=[]):

        """
        fit an active learning strategy
        :param data:
        :param train_index:
        :param snippets:
        :return:
        """
        non_neutral = np.array(train.target) < 2
        selected = np.array(train.index)[non_neutral]
        x = data.bow[selected]
        y = np.array(train.target)[non_neutral]
        self.model.fit(x, y)

        self.current_training = [i for i,n in zip(train.index, non_neutral) if n]
        self.current_training_labels = y.tolist()

        snippets = self._get_snippets(data, train.index)
        labels = []
        for l, s in zip(np.array(train.target), data.sizes[train.index]):
            labels.extend([l] * s)
        self.snippet_model.fit(vstack(snippets), labels)

        return self
