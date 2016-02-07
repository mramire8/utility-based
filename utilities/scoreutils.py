from sklearn.metrics import make_scorer,accuracy_score
import numpy as np


def _prediction_score(y, y_pred):
    return np.mean([y_pred[i][j] for i,j in enumerate(y)])

# def accuracy_score(y, y_pred):
#     return accuracy_score(y, y_pred)


def score_prediction(y, y_pred):
    return make_score_function_proba(_prediction_score)


def make_score_function_proba(fn):
    return make_scorer(fn, greater_is_better=True,needs_proba=True)


def make_score_function(fn):
    return make_scorer(fn, greater_is_better=True,needs_proba=False)


def make_loss_function(fn):
    return make_scorer(fn, greater_is_better=False,needs_proba=True)
