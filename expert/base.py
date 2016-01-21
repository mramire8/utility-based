
class BaseExpert(object):
    """docstring for BaseExpert"""
    def __init__(self, model):
        super(BaseExpert, self).__init__()
        self.oracle = model

    def label(self, data, y=None):
        raise Exception("Expert has not model")        

    def fit(self, data, y=None):
        raise Exception("fit not implemented")