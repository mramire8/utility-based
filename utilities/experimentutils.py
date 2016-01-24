from sklearn.datasets import base as bunch
from learner.strategy import Joint
import numpy as np
from nltk import RegexpTokenizer
from nltk.stem import PorterStemmer



def sample_data(data, train_idx, test_idx):
    sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch())
    
    if len(test_idx) > 0: #if there are test indexes
        sample.train.data = np.array(data.data, dtype=object)[train_idx]
        sample.test.data = np.array(data.data, dtype=object)[test_idx]
        sample.train.target = data.target[train_idx]
        sample.test.target = data.target[test_idx]
        sample.train.bow = data.bow[train_idx]
        sample.test.bow = data.bow[test_idx]
        sample.target_names = data.target_names
        sample.train.remaining = []
    else:
        ## Just shuffle the data
        sample = data
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[train_idx]
        sample.train.data = data_lst
        sample.train.target = data.train.target[train_idx]
        sample.train.bow = data.train.bow[train_idx]
        sample.train.remaining = []
    return sample.train, sample.test

def stemming(doc):

    wnl = PorterStemmer()
    mytokenizer = RegexpTokenizer('\\b\\w+\\b')

    return [wnl.stem(t) for t in mytokenizer.tokenize(doc)]


def get_vectorizer(config):
    limit = config['limit']
    vectorizer = config['vectorizer']

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    if vectorizer == 'tfidf':
        return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))
    elif vectorizer == "tfidfvocab":
        vocab = open(config['vocabulary']).readlines()
        vocab = [v.strip() for v in vocab]
        return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                               vocabulary=vocab)
    elif vectorizer == 'bow':
        from datautils import StemTokenizer
        return CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
                      token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())
    else:
        return None


def get_classifier(cl_name, **kwargs):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    clf = None
    if cl_name is not None:
        if cl_name in "mnb":
            alpha = 1
            if 'parameter' in kwargs:
                alpha = kwargs['parameter']
            clf = MultinomialNB(alpha=alpha)
        elif cl_name == "lr" or cl_name == "lrl1":
            c = 1
            if 'parameter' in kwargs:
                c = kwargs['parameter']
            clf = LogisticRegression(penalty="l1", C=c)
        elif cl_name == "lrl2":
            c = 1
            if 'parameter' in kwargs:
                c = kwargs['parameter']
            clf = LogisticRegression(penalty="l2", C=c)
        else:
            raise ValueError("We need a classifier name for the student [lr|mnb]")
    return clf


def get_learner(learn_config, vct=None, sent_tk=None, seed=None, cost_model=None):
    from learner.base import Learner
    cl_name = learn_config['model']
    clf = get_classifier(cl_name, parameter=learn_config['parameter'])
    learner = Learner(clf)
    if learn_config['type'] == 'utility-based':
        from learner.utility_based import UtilityBasedLearner
        learner = UtilityBasedLearner(clf, snippet_fn=None, utility_fn=None, seed=seed)
    else:
        raise ValueError("We don't know {} leaner".format(learn_config['type']))
    learner.set_loss_function(learn_config['loss_function'])
    learner.set_sent_tokenizer(sent_tk)
    learner.set_vct(vct)
    learner.set_cost_model(cost_model)
    learner.set_cost_fn(get_costfn(learn_config['cost_function']))

    return learner


def get_expert(config, size=None):

    from expert.experts import PredictingExpert, PerfectReluctantDocumentExpert,\
        TrueExpert, NoisyExpert, TrueReluctantExpert

    cl_name = config['model']
    clf = get_classifier(cl_name, parameter=config['parameter'])

    if config['type'] == 'true':
        expert = TrueExpert(None)
    elif config['type'] == 'pred':
        expert = PredictingExpert(clf)
    elif config['type'] == 'noisy':
        p = config['noise_p']
        expert = NoisyExpert(None, p)
    elif config['type'] == 'truereluctant':  # reluctant based on p probability
        p = config['neutral_p']
        expert = TrueReluctantExpert(None, p)
    elif config['type'] == 'perfectreluctant': # reluctant based on unc threshold
        p = config['threshold']
        expert = PerfectReluctantDocumentExpert(clf, p)
    else:
        raise Exception("We don't know {} expert".format(config['type']))

    return expert

def get_bootstrap(config):
    bt = config['bootstrap']
    if 'bootstrap_type' in config:
        mt = config['bootstrap_type']
    else:
        mt = None

    return bt, mt

def get_tokenizer(tk_name, **kwargs):
    if tk_name == 'nltk':
        import nltk
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return sent_detector
    elif tk_name == 'snippet':
        from snippet_tokenizer import SnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = SnippetTokenizer(k=k)
        return sent_detector
    elif tk_name == 'windowsnippet':
        from snippet_tokenizer import WindowSnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = WindowSnippetTokenizer(k=k)
        return sent_detector
    elif tk_name == 'firstksnippet':
        from snippet_tokenizer import FirstWindowSnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = FirstWindowSnippetTokenizer(k=k)
        return sent_detector
    else:
        raise Exception("Unknown sentence tokenizer")


def get_costfn(fn_name):
    import costutils
    return getattr(costutils, fn_name)


def print_file(cost, mean, std, f):
    f.write("COST\tMEAN\tSTDEV\n")
    for a, b, c in zip(cost, mean, std):
        f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\n".format(a, b, c))
    f.close()


def print_cm_file(cost, mean, std, f):
    f.write("COST\tT0\tF1\tF0\tT1\tSTDEV\n")
    for a, b, c in zip(cost, mean, std):
        f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\n".format(a, *b))
    f.close()
