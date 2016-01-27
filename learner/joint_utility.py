import numpy as np
from scipy.sparse import vstack
from copy import copy
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from strategy import StructuredLearner

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

            # parallel = Parallel(n_jobs=2, verbose=True,
            #                     pre_dispatch=4)
            # scores = parallel(delayed(self.evaluation_on_validation_label, check_pickle=False)( lbl,
            #                                 data.bow, tra_x, tra_y, data.validation,data.target[data.validation],i)
            #                  for lbl in labels)
            #
            # uts = sorted(scores, key=lambda x: x[1])

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

    def evaluation_on_validation_label(self, lbl, databow, tra_x, tra_y, val_train, val_target, i):
        x = tra_x
        y = tra_y.append(lbl)
        clf = copy(self.model)
        clf.fit(databow[x], y)
        res = self.evaluation_on_validation(clf, databow[val_train], val_target)
        return (i, lbl, res)

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