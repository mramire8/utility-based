# from strategy import Joint
from joint_utility import Joint
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


class UtilityBasedLearner(Joint):
    """StructuredLearner is the Structured reading implementation """
    def __init__(self, model, snippet_fn=None, utility_fn=None, seed=1):
        super(UtilityBasedLearner, self).__init__(model, snippet_fn=snippet_fn, utility_fn=utility_fn, seed=seed)
        self.loss_fn = self.loss_conditional_error

    def set_utility(self, util):
        self.set_loss_function(util)

    def set_loss_function(self, loss):
        self.loss_fn = getattr(self, loss)

    def loss_conditional_error(self, clf, data, target):
        probs = clf.predict_proba(data)
        loss = np.array([1 - probs[i][j] for i,j in enumerate(target)])
        return loss.mean()

    def loss_error(self, clf, data, target):
        preds = clf.predict(data)
        return accuracy_score(target, preds)

