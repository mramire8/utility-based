
class Learner(object):
    """docstring for Learner"""
    def __init__(self, model, seed=None):
        super(Learner, self).__init__()
        self.model = model
        self.seed = seed
    
    def next_query(self, pool, step):
        raise Exception("Undefined next function")

    def fit(self, X, y):
        raise Exception("Undefined fit function")

    def predict(self,X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def _argmax_x(self, pool, step):
        raise Exception("Undefined objective function")