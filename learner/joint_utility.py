import numpy as np
from scipy.sparse import vstack
from copy import copy
from strategy import StructuredLearner
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from utilities import scoreutils


#######################################################################################################################
class Joint(StructuredLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.current_training = []
        self.current_training_labels = []
        self.minimax = minimax
        self.validation_index = []
        self.loss_fn = None
        self.validation_method = 'eval'
        self.revisiting=False

    def set_minimax(self, minimax):
        if minimax == 'maximize':
            self.minimax = -1
        else:
            self.minimax = 1

    def set_validation_method(self, method):
        self.validation_method = method

    def _subsample_pool(self, rem):
        subpool = list(rem)
        rnd_idx = self.rnd_state.permutation(len(subpool))

        return np.array(subpool)[rnd_idx[:self.subsample]]

    def _select_revisit(self, queries):
        reorder = self.rnd_state.permutation(len(queries))
        return [queries[i][0] for i in reorder[:50]]

    def next_query(self, pool, step):
        """
        Select best candidates for asking the oracle.
        :param pool:
        :param step:
        :return:
        """
        subpool = self._subsample_pool(pool.remaining)

        util1 = self.expected_utility(pool, subpool)  # util list of (utility_score, snipet index of max)
        if self.revisiting:
            revisit = self._select_revisit(pool.revisit)
            rev_util = self.expected_utility(pool, revisit)
            util = util1 + rev_util
            subpool += revisit
        else:
            util = util1

        if self.minimax > 0:  ## minimizing
            max_util = [np.argmin(p) for p in util]  # snippet per document with max/min utility
        else:
            max_util = [np.argmax(p) for p in util]  # snippet per document with max/min utility

        # document with snippet utility max/min
        order = np.argsort([util[i][u] for i, u in enumerate(max_util)])[::self.minimax]

        if self.revisiting:
            index = [a for a in [(subpool[i], max_util[i]) for i in order] if a not in pool.revisit]
        else:
            index = [(subpool[i], max_util[i]) for i in order[:step]]

        return index[:step]

    @staticmethod
    def _average_utilities(utils):

        candidates = len(utils[0])
        folds = len(utils)

        utils = [[(c, l, np.mean([utils[f][c][l][2] for f in range(folds)])) for l in [0,1] ]  for c in range(candidates) ]

        return utils

    def compute_utilities(self, model, data, candidates):
        if self.validation_method == 'cross-validation':
            # raise NotImplementedError("Oops, this experiment should not be tested. Try when it works.")

            tra_y = copy(self.current_training_labels)
            tra_x = copy(self.current_training)

            curr_vals, utils_vals = self._cv_measure(model, data, tra_x, tra_y, candidates, n_folds=4) # clf, data, val_index, candidates,
            curr = np.mean(curr_vals)
            utilities = self._average_utilities(utils_vals)

        else:
            curr = self.current_utility(model, data)
            utilities = self.compute_utility_per_document(data, candidates)

        return curr, utilities

    def expected_utility(self, data, candidates):
        """
        Compute the expected utility of each candidate instance
        :param data: bunch of the pool
        :param candidates: list of candidates
        :return: list of all expected utilities per snippet per document
        """

        curr, utilities = self.compute_utilities(copy(self.model), data, candidates)
        # curr = self.current_utility(copy(self.model), data)
        # utilities = self.compute_utility_per_document(data, candidates)

        snippets = self._get_snippets(data, candidates)
        probs = self._get_snippet_probas(snippets)  # one per snippet
        cost = data.snippet_cost[candidates]
        exp_util = []

        for ut, pr, co in zip(utilities, probs, cost):  # for every document
            exp = []
            for p, c in zip(pr, co):  # for every snippet in the document
                exp.append(np.dot(p[:2], [ut[0][2] - curr, ut[1][2] - curr]) / c)  ## utility over cost, ignore lbl =2

            exp_util.extend([exp])  # one per snippet

        return exp_util

    def _cv_measure(self, clf, data, val_index, val_target, candidates, n_folds=3):

        labels= np.unique(data.target)
        utility = []
        curr_util = []

        cv_indices = val_index
        y_target = np.array(val_target)

        cv = KFold(len(cv_indices), n_folds=n_folds)
        i = 0
        for train, test in cv:

            new_train = cv_indices[train]
            x_train, x_test = data.bow[new_train], data.bow[cv_indices[test]]
            y_train, y_test = y_target[train], y_target[test]

            fold = []

            clf.fit(x_train, y_train)
            curr_util.append(self.loss_fn(clf, x_test, y_test))


            for j, c in enumerate(candidates):
                x_c = vstack([x_train, data.bow[c]])

                doc_util = []
                for lbl in labels:
                    y = np.append(y_train, [lbl])

                    clf.fit(x_c, y)
                    doc_util.append((j, lbl, self.loss_fn(clf, x_test, y_test)))
                    # y = y[:-1]

                fold.append(doc_util)
            utility.append(fold)
            i += 1

        return curr_util, utility



    def current_utility(self, clf, data):

        """
        Compute the current utility of the student model
        :param clf:
        :param data:
        :return:
        """
        tra_y = copy(self.current_training_labels)
        tra_x = copy(self.current_training)

        if self.validation_method == 'cross-validation':
            return self.evaluation_on_validation(clf, data.bow[tra_x], np.array(tra_y), method=self.validation_method)
        else:
            clf.fit(data.bow[tra_x], tra_y)
            return self.evaluation_on_validation(clf, data.validation_set.bow[data.validation],
                                                 np.array(data.validation_set.target), method=self.validation_method)

    def compute_utility_per_document(self, data, candidates):
        """
        For ever document in the candidate list, compute the utility of adding the document ot the training set with every
        possible label.
        :param data:
        :param candidates:
        :return: List of utility per label of each candidate
        """
        labels = self.model.classes_
        tra_y = copy(self.current_training_labels)
        tra_x = copy(self.current_training)
        utilities = []  # two per document
        for i, x_i in enumerate(candidates):
            tra_x.append(x_i)
            uts = [self.evaluation_on_validation_label(lbl, data, tra_x, tra_y, i) for lbl in labels]
            utilities.append(uts)
            # undo training instance
            tra_x = tra_x[:-1]
        return utilities

    def _get_snippet_probas(self, snippets):
        return [self.snippet_model.predict_proba(snip) for snip in snippets]

    def evaluation_on_validation_label(self, lbl, data, tra_x, tra_y, i):

        if lbl < 2:  # if not neutral
            clf = copy(self.model)
            x = tra_x
            y = tra_y + [lbl]
            if self.validation_method == 'cross-validation':
                res = self.evaluation_on_validation(clf, data.bow[x], np.array(y), method=self.validation_method)

            else:

                clf.fit(data.bow[x], y)
                if data.validation_set.bow[data.validation].shape[0] != len(data.validation_set.target):
                    raise Exception("Oops, the validation data has problems")
                res = self.evaluation_on_validation(clf, data.validation_set.bow[data.validation],
                                                    np.array(data.validation_set.target), method=self.validation_method)
            return i, lbl, res
        else:
            # utility of neutral label
            return i, lbl, 0

    def cross_validation_utility(self, clf, data_bow, target, n_folds=10):

        skf = StratifiedKFold(target, n_folds=n_folds)
        scores = []
        for train_index, test_index in skf:
            X_train, X_test = data_bow[train_index], data_bow[test_index]
            y_train, y_test = target[train_index], target[test_index]
            clf.fit(X_train, y_train)
            loss = self.loss_fn(clf, X_test, y_test)
            scores.append(loss)
            # predictions = clf.predict_proba(X_test)
            # scores.extend([predictions[i][j] for i,j in enumerate(y_test)])
        return np.sum(scores)

    def evaluation_on_validation(self, clf, data_bow, target, method=None):
        if method is None:
            raise Exception("Oops, we need the method to compute the score")

        if method == 'cross-validation':
            return self.cross_validation_utility(clf, data_bow, target, n_folds=10)
        else:
            return self.loss_fn(clf, data_bow, target)

    @staticmethod
    def _safe_fit(model, x, y, labels=None):
        lbl = [0, 1]
        if labels is not None:
            lbl = labels
        if hasattr(model, "partial_fit") and False:
            return model.partial_fit(x, y, classes=lbl)
        else:
            return model.fit(x, y)

    def _get_query_snippets(self, data, candidates):

        """
        Get a snippet of each candidate, as seen by the oralce.
        :rtype: tuple
        :param data: bunch with pool candidates
        :param candidates:
        :return: snippets, targets
        """
        if hasattr(data.snippets, "shape"):
            ranges = np.cumsum(data.sizes)
            queries = []
            targets = []
            for di, si, ti in zip(candidates.index, candidates.snip, candidates.target):
                if si is not None:
                    queries.append(data.snippets[0 if di == 0 else ranges[di - 1]:ranges[di]][si])
                    targets.append(ti)
                else:
                    snippets = data.snippets[0 if di == 0 else ranges[di - 1]:ranges[di]]
                    queries.extend(snippets)
                    targets.extend([ti] * len(snippets))
        else:
            # queries = [data.snippets[d][np.ix_([s])] for d,s in zip(candidates.index, candidates.snip)]
            queries = []
            targets = []
            for di, si, ti in zip(candidates.index, candidates.snip, candidates.target):
                if si is not None:
                    queries.append(data.snippets[di][si])
                    targets.append(ti)
                else:
                    snippets = data.snippets[di]
                    queries.append(snippets)
                    if hasattr(snippets, "shape"):
                        n = snippets.shape[0]
                    else:
                        n = len(snippets)
                    targets.extend([ti] * n)

        return queries, targets

    def _get_query_snippets_v0(self, data, candidates):

        """
        Get all snippets of each candidate, all possible snippets (according to the snippet function) for each document
        :param data:
        :param candidates:
        :return:
        """
        if hasattr(data.snippets, "shape"):
            ranges = np.cumsum(data.sizes)
            queries = []
            targets = []
            for di, si, ti in zip(candidates.index, candidates.snip, candidates.target):
                snippets = data.snippets[0 if di == 0 else ranges[di - 1]:ranges[di]]
                queries.extend(snippets)
                targets.extend([ti] * len(snippets))
        else:
            # queries = [data.snippets[d][np.ix_([s])] for d,s in zip(candidates.index, candidates.snip)]
            queries = []
            targets = []
            for di, si, ti in zip(candidates.index, candidates.snip, candidates.target):
                snippets = data.snippets[di]
                queries.append(snippets)
                if hasattr(snippets, "shape"):
                    n = snippets.shape[0]
                else:
                    n = len(snippets)
                targets.extend([ti] * n)

        return queries, targets

    def _get_query_labels(self, target, candidates):
        pass

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

        self.model = self._safe_fit(self.model, x, y)

        self.current_training = [i for i, n in zip(train.index, non_neutral) if n]
        self.current_training_labels = y.tolist()

        snippets, labels = self._get_query_snippets(data, train)
        # labels = []
        # for l, s in zip(np.array(train.target), data.sizes[train.index]):
        #     labels.extend([l] * s)

        self.snippet_model = self._safe_fit(self.snippet_model, vstack(snippets), labels, labels=[0, 1, 2])

        return self


#######################################################################################################################

class JointUncertainty(Joint):
    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(JointUncertainty, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                               minimax=minimax, seed=seed)

    def compute_utility_per_document(self, data, candidates):
        probs = self.model.predict_proba(data.bow[candidates])
        unc = 1 - probs.max(axis=1)
        return unc

    def expected_utility(self, data, candidates):
        """
        Compute the expected utility of each candidate instance
        :param data: bunch of the pool
        :param candidates: list of candidates
        :return: list of all expected utilities per snippet per document
        """
        utilities = self.compute_utility_per_document(data, candidates)

        cost = data.snippet_cost[candidates]

        exp_util = [np.dot(c, u) for u, c in zip(utilities, cost)]

        return exp_util


#######################################################################################################################
# Module functions
def _evaluation_on_validation_per_label(model, lbl, data, tra_x, tra_y, i, function):
    if lbl < 2:  # if not neutral
        x = tra_x
        y = tra_y + [lbl]
        clf = copy(model)
        clf.fit(data.bow[x], y)
        res = _evaluation_on_validation(clf, data.validation_set.bow[data.validation],
                                        data.validation_set.target[data.validation], function)
        return (i, lbl, res)
    else:
        # utility of neutral label
        return (i, lbl, 0)


def _evaluation_on_validation(clf, data, target, function):
    return function(clf, data, target)
