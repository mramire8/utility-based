__author__ = 'maru'

from experts import PredictingExpert
import numpy as np
import nltk

class NoisyReluctantDocumentExpert(PredictingExpert):

    def __init__(self, oracle, reluctant_threshold, factor=1., data_size=None, seed=43212):
        super(NoisyReluctantDocumentExpert, self).__init__(oracle)
        self.reluctant_threhold = reluctant_threshold
        self.rnd = np.random.RandomState(seed)
        self.scale_factor = factor
        self.coin = None
        if data_size is not None:
            self.coin = self.rnd.random_sample((data_size, len(range(1,6))))
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def set_scale_factor(self, factor):
        self.scale_factor = factor

    def get_scale_factor(self):
        return self.scale_factor

    def _k_to_col(self, k):
        '''
        convert from query size to matrix column. To use the coin flip matrix
        :param k: instance size = [10, 25, 50, 75, 100]
        :return: int \in [0,4]
        '''
        col = range(1,6)
        return min(col.index(k), len(col)-1)
        # return col.index(k)

    def _flip_coin(self, x, y):
        '''
        return a random number for instance in position x,y in the coin matrix
        :param x:
        :param y:
        :return:
        '''
        if x == None:
            coin = self.rnd.random_sample()
        else:
            y = min(y, self.coin.shape[1]-1)
            coin = self.coin[x, y]
        return coin

    def fit(self, X_text, y=None, vct=None):
        sx = vct.transform(X_text)
        self.oracle.fit(sx, y)
        return self

    def label(self, query, y=None, index=None, size=None):
        data = query
        if isinstance(query, dict):
            data = query.bow
        # text, target, index, snipet, bow
        # Initizalize with all neutrals
        prediction = np.array([None] * data.shape[0], dtype=object)

        # Compute the uncertainty of the oracle
        proba = self.oracle.predict_proba(data)
        unc = 1. - proba.max(axis=1)

        # Get remaining true labels
        prediction[unc < self.reluctant_threhold] = y[unc < self.reluctant_threhold]

        # Get the conditional error, for accuracy probability
        scaled_ce = unc * self.get_scale_factor()

        # Get size in number of sentences
        sents = self.sent_detector.tokenize_sents(query.snippet)
        size = [len(s) for s in sents]

        # Flip a coin for every instance
        coin_flipped = [self._flip_coin(x, s) for x, s in zip(query.index, size)]

        for i, p in enumerate(prediction):
            if p is not None:  # If not neutral
                if coin_flipped[i] <= scaled_ce[i]:
                    prediction[i] = 1 - prediction[i]

        return prediction
