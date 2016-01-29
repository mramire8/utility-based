__author__ = 'mramire8'

import numpy as np


def unit_cost(query, cost_model=None):
    n = 1
    if isinstance(query, dict):
        n = len(query.snippet)
    else:
        n = len(query)
    return [1.] * n


def word_cost(query, cost_model=None):
    c = 0
    if isinstance(query, int):
        return query

    x_text = query
    if isinstance(x_text, basestring):
        c = len(x_text.split())
    elif isinstance(x_text, list) or isinstance(x_text, np.array):
        if x_text is not None:
            c = [len(x.split()) for x in x_text]

    return c


def log_word_cost(query, cost_model=None):
    return np.log2(1+word_cost(query, cost_model=cost_model))


def log_linear_cost(query, cost_model=None):
    words = word_cost(query, cost_model=cost_model)
    if words <= 1:
        return 4.
    else:
        return 5*np.log(words) - 7


def intra_cost(query, cost_model=None):
    if cost_model is None:
        raise ValueError("Cost model is not available.")
    c = 0
    kvals = sorted(cost_model.keys())
    cost = [cost_model[k] for k in kvals]

    if isinstance(query, dict):
        x_text = query.snippet
    else:
        x_text = query
    if x_text is not None:
        c = [_cost_intrapolated(len(x.split()), cost, kvals) for x in x_text]

    return c


def _cost_intrapolated(x, cost, kvalues):
    import numpy as np

    binx = min(np.digitize([x], kvalues)[0], len(cost) - 1)
    lbbinx = max(binx - 1, 0)

    y1 = cost[lbbinx] if lbbinx >= 0  else 0
    y2 = cost[binx]
    x1 = kvalues[lbbinx] if lbbinx >= 0 else 0
    x2 = kvalues[binx]

    if (x2 - x1) == 0:
        return cost[0]

    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2

    if x < kvalues[0]:
        y = cost[0]
    elif x > kvalues[-1]:
        y = cost[-1]
    else:
        y = (m * x) + b
    return y
