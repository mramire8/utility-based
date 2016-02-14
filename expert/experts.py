from base import BaseExpert
import numpy as np
from scipy.sparse import vstack

class TrueExpert(BaseExpert):
    """docstring for TrueExpert"""

    def __init__(self, oracle):
        super(TrueExpert, self).__init__(oracle)

    def label(self, data, y=None):
        if y is None:
            raise Exception("True labels are missing")
        else:
            return y

    def fit(self, X, y=None, vct=None):
        return self


class NoisyExpert(BaseExpert):

    def __init__(self, oracle, noise_p, seed=8273645):

        super(NoisyExpert, self).__init__(oracle)
        self.noise_p = noise_p
        self.rnd = np.random.RandomState(seed)

    def label(self, data, y=None):

        if len(y) == 1:
            coin = self.rnd.random_sample()

            if coin < self.noise_p:
                return 1 - y
            else:
                return y
        else:
            coin = self.rnd.random_sample(y.shape)
            new_target = y.copy()
            new_target[coin < self.noise_p] = 1 - new_target[coin < self.noise_p]
            return new_target

    def fit(self, X, y=None, vct=None):
        return self


class PredictingExpert(BaseExpert):
    """docstring for PredictingExpert"""

    def __init__(self, oracle):
        super(PredictingExpert, self).__init__(oracle)

    def label(self, data, y=None):
        return self.oracle.predict(data)

    def fit(self, data, y=None, vct=None):
        if y is not None:
            self.oracle.fit(data.bow, y)
        else:
            self.oracle.fit(data.bow, data.target)
        return self


class ReluctantDocumentExpert(PredictingExpert):

    def __init__(self, oracle, reluctant_threshold, seed=43212):
        super(ReluctantDocumentExpert, self).__init__(oracle)
        self.reluctant_threhold = reluctant_threshold
        self.rnd = np.random.RandomState(seed)
        self.neutral_value = 2

    def fit(self, X_text, y=None, vct=None):
        sx = vct.transform(X_text)
        self.oracle.fit(sx, y)
        return self

    def label(self, data, y=None):
        prediction = np.array([self.neutral_value] * data.shape[0], dtype=object)
        proba = self.oracle.predict_proba(data)
        pred = self.oracle.predict(data)
        unc = proba.min(axis=1)
        prediction[unc < self.reluctant_threhold] = pred[unc < self.reluctant_threhold]

        return prediction


class PerfectReluctantDocumentExpert(PredictingExpert):

    def __init__(self, oracle, reluctant_threshold, tokenizer=None, seed=43212):
        super(PerfectReluctantDocumentExpert, self).__init__(oracle)
        self.reluctant_threhold = reluctant_threshold
        self.rnd = np.random.RandomState(seed)
        self.neutral_value = 2

    def fit(self, x, y=None, vct=None):

        self.oracle.fit(x, y)
        return self

    def label(self, query, y=None, **kwargs):
        data = query
        if isinstance(query, dict):
            data = query.bow

        if isinstance(data, list):
            n = len(data)
        else:
            n = data.shape[0]

        prediction = np.array([self.neutral_value] * n)
        data = vstack(data)
        proba = self.oracle.predict_proba(data)
        unc = 1. - proba.max(axis=1)

        if sum(unc < self.reluctant_threhold) > 0:
            prediction[unc < self.reluctant_threhold] = y[unc < self.reluctant_threhold]

        return prediction


class TrueReluctantExpert(TrueExpert):

    def __init__(self, oracle, reluctant_p,  seed=43212):
        super(TrueReluctantExpert, self).__init__(oracle)
        self.rnd = np.random.RandomState(seed)
        self.reluctant_p = reluctant_p
        self.neutral_value = 2

    def label(self, data, y=None):

        prediction = np.array(y, dtype=object)  # copy the true labels

        coin = self.rnd.random_sample(len(y))  # flip a coin for neutral probability

        prediction[coin < self.reluctant_p] = self.neutral_value  ## if coin is < p then is neutral

        return prediction