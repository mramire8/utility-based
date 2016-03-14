# from strategy import Joint
from joint_utility import Joint
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class UtilityBasedLearner(Joint):
    """StructuredLearner is the Structured reading implementation """
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1, minimax=-1):
        super(UtilityBasedLearner, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                                  seed=seed, minimax=minimax)
        self.loss_fn = self.loss_conditional_error

    def set_utility(self, util):
        self.set_loss_function(util)

    def set_loss_function(self, loss):
        self.loss_fn = getattr(self, loss)

    def loss_conditional_error(self, clf, data, target):
        probs = clf.predict_proba(data)
        loss = np.array([1 - probs[i][j] for i,j in enumerate(target)])
        return loss.mean()

    def loss_prediction(self, clf, data, target):
        probs = clf.predict_proba(data)
        loss = np.array([probs[i][j] for i,j in enumerate(target)])
        return loss.mean()

    def loss_error(self, clf, data, target):
        preds = clf.predict(data)
        return accuracy_score(target, preds)


#######################################################################################################################
class FirstK(UtilityBasedLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(FirstK, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn,
                                     seed=seed, minimax=minimax)
        raise Exception("This should compute a utility as well for the documents and reorder them")

    def next_query(self, pool, step):

        """
        Select the first snippet of every instance in the pool.
        :param pool:
        :param step:
        :return:
        """
        subpool = self._subsample_pool(pool.remaining)

        index = [(s, 1) for s in subpool[:step]]
        return index

#######################################################################################################################
class RandomK(UtilityBasedLearner):
    """docstring for randomk"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1):
        super(RandomK, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)

    def next_query(self, pool, step):

        """
        Select the first snippet of every instance in the pool.
        :param pool:
        :param step:
        :return:
        """
        subpool = self._subsample_pool(pool.remaining)
        rnd_num = self._snippet_rnd(step)
        index = [(s, 1) for s in zip(subpool[:step],rnd_num)]
        return index


#######################################################################################################################
class JointCheat(UtilityBasedLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1, snip_model=None, neutral=.4):
        super(JointCheat, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.snippet_model = snip_model
        self.neutral_threshold = neutral

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

        return self

    def set_snippet_model(self, clf):
        self.snippet_model = clf

    def _get_snippet_probas(self, snippets):
        def transform_label(p, neutral_threshold):
            lbl = [0] * 3
            if p.min() < neutral_threshold:
                lbl[p.argmax()] = 1
            else:
                lbl = np.array([0,0,1])
            return np.array(lbl)

        probs = [self.snippet_model.predict_proba(snip) for snip in snippets]

        corrected = [[transform_label(p,self.neutral_threshold) for p in ps] for ps in probs]

        return corrected

#######################################################################################################################
class JointNoisyCheat(UtilityBasedLearner):
    """docstring for Joint"""

    def __init__(self, model, snippet_fn=None, utility_fn=None, minimax=-1, seed=1, snip_model=None, neutral=.4):
        super(JointNoisyCheat, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.snippet_model = snip_model
        self.neutral_threshold=neutral

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

        return self

    def set_snippet_model(self, clf):
        self.snippet_model = clf

    def _get_snippet_probas(self, snippets):
        def transform_label_v1(p,neutral_threshold):
            lbl = [0] * 3
            if p.min() < neutral_threshold:
                lbl[p.argmax()] = p.max()
                lbl[1-p.argmax()] = 1-p.max()
            else:
                lbl = np.array([0,0,1])
            return np.array(lbl)

        def transform_label(p, neutral_threshold):
            lbl = [0] * 3
            if p.min() < neutral_threshold:
                lbl[p.argmax()] = 1
            else:
                lbl = np.array([0,0,1])
            return np.array(lbl)

        probs = [self.snippet_model.predict_proba(snip) for snip in snippets]

        corrected = [[transform_label(p,self.neutral_threshold) for p in ps] for ps in probs]

        return corrected
