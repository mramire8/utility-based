import numpy as np

def loss_conditional_error( clf, data, target):
    probs = clf.predict_proba(data)
    loss = np.array([1 - probs[i][j] for i,j in enumerate(target)])
    return loss.mean()