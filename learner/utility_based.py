from strategy import StructuredLearner


class UtilityBasedLearner(StructuredLearner):
    """StructuredLearner is the Structured reading implementation """
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(UtilityBasedLearner, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)

    def _subsample_pool(self, X):
        pass

    def next(self, pool, step):

        #_subsample_pool
        #compute expected utility
        # select max expected utility
        pass
