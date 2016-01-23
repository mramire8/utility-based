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

    # @staticmethod
    # def convert_to_sentence(X_text, y, sent_tk, limit=None):
    #     """
    #     >>> import nltk
    #     >>> sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
    #     >>> print StructuredLearner.convert_to_sentence(['hi there. you rock. y!'], [1], sent_tk, limit=2)
    #     (['hi there.', 'you rock.'], [1, 1])
    #     """
    #     sent_train = []
    #     labels = []
    #
    #     ## Convert the documents into sentences: train
    #     # for t, sentences in zip(y, sent_tk.batch_tokenize(X_text)):
    #     for t, sentences in zip(y, sent_tk.tokenize_sents(X_text)):
    #         if limit > 0:
    #             sents = [s for s in sentences if len(s.strip()) > limit]
    #         elif limit == 0 or limit is None:
    #             sents = [s for s in sentences]
    #
    #         sent_train.extend(sents)  # at the sentences separately as individual documents
    #
    #         labels.extend(StructuredLearner._get_sentence_labels(t,len(sents)))  # Give the label of the document to all its sentences
    #
    #     return sent_train, labels  # , dump
    #
    # @staticmethod
    # def _get_sentence_labels(t, n):
    #     '''
    #     For a set of sentences get the corresponding labels
    #     :param t:
    #     :param n:
    #     :return:
    #     '''
    #     if isinstance(t, list):
    #         # the sentences are labeled individually
    #         if len(t) == n:
    #             return t
    #         else:
    #             raise Exception("Error: number of labels should be %s" % n)
    #     else:
    #         # only the document label is available, propagate to sentences
    #         return [t] * n

    def get_name(self):
        return "{}{}".format(self.utility.__name__, self.snippet_utility.__name__)

    def fit(self, X, y, doc_text=None, limit=None):
        # fit student
        self.model.fit(X, y)

        #fit sentence
        # sx, sy = self.convert_to_sentence(doc_text, y, self.sent_tokenizer, limit=limit)
        # sx = self.vct.transform(sx)
        # self.snippet_model.fit(sx, sy)

        return self

    # def _utility_rnd(self, X):
    #     if X.shape[0] == 1:
    #         return self.rnd_state.random_sample()
    #     else:
    #         return self.rnd_state.random_sample(X.shape[0])
    #
    # def _utility_unc(self, X):
    #     p = self.model.predict_proba(X)
    #     if X.shape[0] == 1:
    #         return 1. - p.max()
    #     else:
    #         return 1. - p.max(axis=1)

    def _subsample_pool(self, X):
        raise NotImplementedError("Implement in the subclass")

    def _compute_utility(self, X):
        return self.utility(X)

    # def to_matrix(self, snip_bows):
    #     data= snip_bows[0]
    #     for s in snip_bows[1:]:
    #         data = vstack([data, s], format='csr')
    #     return data

    def _query(self, pool, snippets, indices, snippet_index, bow=None):
        pass

    # def _get_target(self, targets, indices):
    #     sent_target = []
    #     for t, i in zip(targets, indices):
    #         if isinstance(t, list):
    #             sent_target.append(t[i])
    #         else:
    #             sent_target.append(t)
    #
    #     return np.array(sent_target)

    def set_utility(self, util):
        self.utility = getattr(self, util)

    def set_snippet_utility(self, util):
        self.snippet_utility = getattr(self, util)

    def set_sent_tokenizer(self, tokenizer):
        self.sent_tokenizer = tokenizer

    def set_vct(self, vct):
        self.vct = vct

    ## SNIPPET UTILITY FUNCTIONS
    # def _snippet_max(self, X):
    #     p = self.snippet_model.predict_proba(X)
    #     if X.shape[0] == 1:
    #         return p.max()
    #     else:
    #         return p.max(axis=1)
    #
    def _snippet_rnd(self, X):
        return self.sent_rnd.random_sample(X.shape[0])
    #
    # def _snippet_first(self, X):
    #     n = X.shape[0]
    #     scores = np.zeros(n)
    #     scores[0] = 1
    #     return scores
    #
    # def _snippet_cost(self, snips):
    #     # return np.array([self.cost_fn(xi, cost_model=self.cost_model) for xi in snips])
    #     return np.array(self.cost_fn(snips, cost_model=self.cost_model))
    #
    # def _create_matrix(self, x_sent, x_len):
    #
    #     X = np.zeros((len(x_sent), x_len))
    #
    #     return X
    #
    # def _get_sentences(self, x_text):
    #     text = self.sent_tokenizer.tokenize_sents(x_text)
    #     text_min = []
    #     for sentences in text:
    #         text_min.append([s for s in sentences if len(s.strip()) > 2])  # at least 2 characters
    #     return text_min
    #

    def _get_snippets(self, data, candidates):
        ranges = np.cumsum(data.sizes)
    #     print 0 if i==0 else ranges[i-1],ranges[i]
        snips = []
        for i in candidates:
            snips.append(data.snippets[0 if i==0 else ranges[i-1]:ranges[i]])
        return snips

    def _get_probs_per_snippet(self, data, candidates):
        ranges = np.cumsum(data.sizes)
    #     print 0 if i==0 else ranges[i-1],ranges[i]
        snips = []
        for i in candidates:
            snips.append(data.snippets[0 if i==0 else ranges[i-1]:ranges[i]])
        return snips

    # def snippet_roi(self, s, s_text):
    #     return self.snippet_utility(s) / self._snippet_cost(s_text)

    def _compute_snippet(self, pool):
        """select the snippet with the best score for each document"""
        # # scores = super(Joint, self)._compute_snippet(x_text)
        # import sys
        x_sent_bow, x_sent, x_len = self._get_snippets(pool) # a matrix of all snippets (stacked), and the size

        snip_prob = self.snippet_model.predict_proba(x_sent_bow)

        #
        # x_scores = self._create_matrix(x_sent, x_len)
        # y_pred = self._create_matrix(x_sent, x_len)
        #
        # for i, s in enumerate(x_sent_bow):
        #     score_i = np.ones(x_len) * -1 * sys.maxint
        #     y_pred_i = np.zeros(x_len)
        #
        #     score_i[:s.shape[0]] = self.snippet_roi(s, x_sent[i])
        #     y_pred_i[:s.shape[0]] = self.snippet_model.predict(s) + 1  # add 1 to avoid prediction 0, keep the sparsity
        #     x_scores[i] = score_i
        #     y_pred[i] = y_pred_i
        #
        # x_scores = self._do_calibration(x_scores, y_pred)
        #
        # # Note: this works only if the max score is always > 0
        # sent_index = x_scores.argmax(axis=1)   # within each document the sentence with the max score
        #
        # sent_max = x_scores.max(axis=1)   # within each document the sentence with the max score
        # sent_text = [x_sent[i][maxx] for i, maxx in enumerate(sent_index)]
        # sent_text = np.array(sent_text, dtype=object)
        # sent_bow = np.array([x_sent_bow[i][maxx] for i,maxx in enumerate(sent_index)], dtype=object)
        #
        # return sent_max, sent_text, sent_index, sent_bow
        pass

    def __str__(self):
        return "{}(model={}, snippet_model={}, utility={}, snippet={})".format(self.__class__.__name__, self.model,
                                                                               self.snippet_model, self.utility,
                                                                               self.snippet_utility)


class Joint(StructuredLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1,  seed=1):
        super(Joint, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.current_training = []
        self.current_training_labels = []
        self.minimax = minimax
        self.validation_index=[]
        self.loss_fn = None

    def set_minimax(self, minimax):
        if minimax =='maximize':
            self.minimax = -1
        else:
            self.minimax = 1
    
    def _subsample_pool(self, rem):
        subpool = list(rem)
        self.rnd_state.shuffle(subpool)
        subpool = subpool[:250]

        return subpool

    
    def next(self, pool, step):

        subpool = self._subsample_pool(pool.remaining)

        util = self.expected_utility(pool, subpool) # util list of (utility_score, snipet index of max)

        max_util = [np.argsort(p)[::self.minimax][0] for p in util]  # snippet per document with max/min utility

        order = np.argsort([util[i][u] for i,u in enumerate(max_util)])[::self.minimax]  # document with snippet utlity max

        index = [(subpool[i], max_util[i]) for i in order[:step]]

        #build the query
        # query = self._query(pool, snippet_text[order][:step], index, sent_index[order][:step], bow=sent_bow[order][:step])
        
        return index


    def expected_utility(self, data, candidates):
        labels = self.model.classes 
        tra_y = self.current_training_labels
        tra_x = self.current_training
        clf  = copy(self.model)
        utilities = []  # two per document
        for i, x_i in enumerate(candidates):
            tra_x.append(x_i)
            uts = []

            for lbl in labels:
                tra_y.append(lbl)
                clf.fit(data.bow[tra_x], tra_y)
                u = self.evaluation_on_validation(clf, data.bow[data.validation])
                uts.append((i, lbl, u))
                # undo labels
                tra_y = tra_y[:-1]

            utilities.append(uts)

            # undo training instance
            tra_x = tra_x[:-1]

        snippets = self._get_snippets(data, candidates)
        probs = [self.snippet_model.predict_proba(snip) for snip in snippets] # one per snippet

        exp_util = []

        for ut, pr in zip(utilities, probs):

            exp  = pr[:, 0] * ut[0][2]  + pr[:, 1] * ut[1][2]
            exp_util.extend([exp]) # one per snippet

        return exp_util


    def compute_utility(self, x, current_training):
        # copy current training data

        # for every label:
            # add add x and add label
            # copy clf
            # compute utility/loss on validations data with clf_copy

        # undo
        pass


    def evaluation_on_validation(self, clf, data):
        loss = self.loss_fn(clf, data)

        return loss

    def fit(self, X, y, doc_text=None, limit=None, train_index=[]):

        super(Joint, self).fit(X, y, doc_text=doc_text, limit=limit)
        self.current_training.extend(train_index)
        self.current_training_labels.extend(y)
        return self

